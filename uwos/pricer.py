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

from uwos.schwab_auth import (
    SchwabAuthConfig,
    SchwabLiveDataService,
    compact_occ_to_schwab_symbol,
)

BASE_REQUIRED_COLUMNS = [
    "ticker",
    "strategy",
    "expiry",
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


def validate_shortlist_columns(df: pd.DataFrame) -> None:
    missing = [col for col in BASE_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Shortlist CSV missing required columns: {missing}")

    for _, row in df.iterrows():
        strategy = str(row.get("strategy", "")).strip()
        if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            needed = ["short_put_leg", "long_put_leg", "short_call_leg", "long_call_leg"]
        else:
            needed = ["short_leg", "long_leg"]
        missing_row = [c for c in needed if not str(row.get(c, "")).strip()]
        if missing_row:
            ticker = str(row.get("ticker", "")).strip()
            raise ValueError(
                f"Shortlist CSV missing required leg columns for {strategy or 'UNKNOWN'} {ticker}: {missing_row}"
            )


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
    parser.add_argument(
        "--snapshot-out-json",
        default="",
        help="Optional path to write snapshot metadata (chain status/query symbols/errors).",
    )
    parser.add_argument(
        "--hard-invalidation",
        action="store_true",
        help="If set, invalidation breach is treated as hard live failure (default: warning only).",
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
    snapshot_out_json = (
        Path(args.snapshot_out_json).expanduser().resolve() if args.snapshot_out_json else None
    )

    df = pd.read_csv(shortlist_csv)
    validate_shortlist_columns(df)

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
    for col in ["short_leg", "long_leg", "short_put_leg", "long_put_leg", "short_call_leg", "long_call_leg"]:
        if col not in df.columns:
            df[col] = ""
    df["short_leg_live_symbol"] = df["short_leg"].map(convert_leg_symbol)
    df["long_leg_live_symbol"] = df["long_leg"].map(convert_leg_symbol)
    df["short_put_leg_live_symbol"] = df["short_put_leg"].map(convert_leg_symbol)
    df["long_put_leg_live_symbol"] = df["long_put_leg"].map(convert_leg_symbol)
    df["short_call_leg_live_symbol"] = df["short_call_leg"].map(convert_leg_symbol)
    df["long_call_leg_live_symbol"] = df["long_call_leg"].map(convert_leg_symbol)

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

    leg_symbols = []
    for col in [
        "short_leg_live_symbol",
        "long_leg_live_symbol",
        "short_put_leg_live_symbol",
        "long_put_leg_live_symbol",
        "short_call_leg_live_symbol",
        "long_call_leg_live_symbol",
    ]:
        leg_symbols.extend(df[col].dropna().astype(str).tolist())
    leg_symbols = [sym for sym in leg_symbols if sym]
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

    def leg_payload(symbol: Any) -> Dict[str, Any]:
        sym = str(symbol or "").strip()
        return option_quotes.get(sym, {}) if sym else {}

    def leg_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        quote = payload.get("quote", payload)
        ref = payload.get("reference", {})
        return {
            "bid": safe_float(quote.get("bidPrice")),
            "ask": safe_float(quote.get("askPrice")),
            "last": safe_float(quote.get("lastPrice")),
            "mark": safe_float(quote.get("mark")),
            "delta": safe_float(quote.get("delta")),
            "oi": safe_float(quote.get("openInterest")),
            "volume": safe_float(quote.get("totalVolume")),
            "strike": safe_float(ref.get("strikePrice")),
            "expiry": parse_reference_expiry(ref),
        }

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        ticker = str(row.get("ticker", "")).strip().upper()
        strategy = str(row.get("strategy", "")).strip()
        net_type = str(row.get("net_type", "")).strip().lower()
        entry_op, entry_threshold, entry_unit = parse_entry_gate(row.get("entry_gate"))
        invalidation_op, invalidation_level = parse_invalidation_rule(row.get("invalidation"))
        width_row = safe_float(row.get("width"))

        chain_status = chain_status_by_ticker.get(ticker, "UNKNOWN")
        chain_syms = chain_symbols_by_ticker.get(ticker, set())

        short_live_symbol = str(row.get("short_leg_live_symbol", "")).strip()
        long_live_symbol = str(row.get("long_leg_live_symbol", "")).strip()
        short_call_live_symbol = str(row.get("short_call_leg_live_symbol", "")).strip()
        long_call_live_symbol = str(row.get("long_call_leg_live_symbol", "")).strip()

        # Defaults for optional 4-leg paths.
        short_put = {}
        long_put = {}
        short_call = {}
        long_call = {}
        short_in_chain = None
        long_in_chain = None
        short_call_in_chain = None
        long_call_in_chain = None

        if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            short_put = leg_fields(leg_payload(short_live_symbol))
            long_put = leg_fields(leg_payload(long_live_symbol))
            short_call = leg_fields(leg_payload(short_call_live_symbol))
            long_call = leg_fields(leg_payload(long_call_live_symbol))

            if chain_status == "SUCCESS":
                short_in_chain = short_live_symbol in chain_syms if short_live_symbol else None
                long_in_chain = long_live_symbol in chain_syms if long_live_symbol else None
                short_call_in_chain = short_call_live_symbol in chain_syms if short_call_live_symbol else None
                long_call_in_chain = long_call_live_symbol in chain_syms if long_call_live_symbol else None

            condor_net_type = "debit" if strategy == "Long Iron Condor" else "credit"
            put_net_bid_ask, put_net_mark = compute_live_net(
                net_type=condor_net_type,
                short_bid=short_put.get("bid"),
                short_ask=short_put.get("ask"),
                short_mark=short_put.get("mark"),
                long_bid=long_put.get("bid"),
                long_ask=long_put.get("ask"),
                long_mark=long_put.get("mark"),
            )
            call_net_bid_ask, call_net_mark = compute_live_net(
                net_type=condor_net_type,
                short_bid=short_call.get("bid"),
                short_ask=short_call.get("ask"),
                short_mark=short_call.get("mark"),
                long_bid=long_call.get("bid"),
                long_ask=long_call.get("ask"),
                long_mark=long_call.get("mark"),
            )
            live_net_bid_ask = (
                put_net_bid_ask + call_net_bid_ask
                if (put_net_bid_ask is not None and call_net_bid_ask is not None)
                else None
            )
            live_net_mark = (
                put_net_mark + call_net_mark
                if (put_net_mark is not None and call_net_mark is not None)
                else None
            )

            put_width_live = (
                abs(short_put["strike"] - long_put["strike"])
                if (short_put.get("strike") is not None and long_put.get("strike") is not None)
                else None
            )
            call_width_live = (
                abs(long_call["strike"] - short_call["strike"])
                if (short_call.get("strike") is not None and long_call.get("strike") is not None)
                else None
            )
            width_live = (
                max(put_width_live, call_width_live)
                if (put_width_live is not None and call_width_live is not None)
                else None
            )
            width_for_risk = width_live if width_live is not None else width_row
        else:
            short_data = leg_fields(leg_payload(short_live_symbol))
            long_data = leg_fields(leg_payload(long_live_symbol))
            short_put = short_data
            long_put = long_data
            live_net_bid_ask, live_net_mark = compute_live_net(
                net_type=net_type,
                short_bid=short_data.get("bid"),
                short_ask=short_data.get("ask"),
                short_mark=short_data.get("mark"),
                long_bid=long_data.get("bid"),
                long_ask=long_data.get("ask"),
                long_mark=long_data.get("mark"),
            )
            if chain_status == "SUCCESS":
                short_in_chain = short_live_symbol in chain_syms if short_live_symbol else None
                long_in_chain = long_live_symbol in chain_syms if long_live_symbol else None
            width_live = (
                abs(short_data["strike"] - long_data["strike"])
                if (short_data.get("strike") is not None and long_data.get("strike") is not None)
                else None
            )
            width_for_risk = width_live if width_live is not None else width_row

        gate_pass_live = None
        if entry_threshold is not None and live_net_bid_ask is not None:
            gate_eps = 1e-9
            if net_type == "credit":
                gate_pass_live = live_net_bid_ask >= (entry_threshold - gate_eps)
            else:
                gate_pass_live = live_net_bid_ask <= (entry_threshold + gate_eps)

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

        entry_structure_ok = True
        entry_structure_reason = ""
        if spot_eval_live is not None and math.isfinite(spot_eval_live):
            if strategy == "Bull Put Credit":
                short_put_strike = safe_float(short_put.get("strike"))
                long_put_strike = safe_float(long_put.get("strike"))
                reasons = []
                if (
                    short_put_strike is not None
                    and math.isfinite(short_put_strike)
                    and long_put_strike is not None
                    and math.isfinite(long_put_strike)
                    and not (long_put_strike < short_put_strike < spot_eval_live)
                ):
                    reasons.append(
                        f"need long_put < short_put < spot (got {long_put_strike:.2f} < {short_put_strike:.2f} < {spot_eval_live:.2f})"
                    )
                if (
                    live_net_bid_ask is not None
                    and math.isfinite(live_net_bid_ask)
                    and short_put_strike is not None
                    and math.isfinite(short_put_strike)
                ):
                    be = short_put_strike - live_net_bid_ask
                    if spot_eval_live <= be:
                        reasons.append(
                            f"spot {spot_eval_live:.2f} at/below breakeven {be:.2f}"
                        )
                if reasons:
                    entry_structure_ok = False
                    entry_structure_reason = "; ".join(reasons)
            elif strategy == "Bear Call Credit":
                short_call_strike = safe_float(short_put.get("strike"))
                long_call_strike = safe_float(long_put.get("strike"))
                reasons = []
                if (
                    short_call_strike is not None
                    and math.isfinite(short_call_strike)
                    and long_call_strike is not None
                    and math.isfinite(long_call_strike)
                    and not (spot_eval_live < short_call_strike < long_call_strike)
                ):
                    reasons.append(
                        f"need spot < short_call < long_call (got {spot_eval_live:.2f} < {short_call_strike:.2f} < {long_call_strike:.2f})"
                    )
                if (
                    live_net_bid_ask is not None
                    and math.isfinite(live_net_bid_ask)
                    and short_call_strike is not None
                    and math.isfinite(short_call_strike)
                ):
                    be = short_call_strike + live_net_bid_ask
                    if spot_eval_live >= be:
                        reasons.append(
                            f"spot {spot_eval_live:.2f} at/above breakeven {be:.2f}"
                        )
                if reasons:
                    entry_structure_ok = False
                    entry_structure_reason = "; ".join(reasons)
            elif strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                long_put_strike = safe_float(long_put.get("strike"))
                short_put_strike = safe_float(short_put.get("strike"))
                short_call_strike = safe_float(short_call.get("strike"))
                long_call_strike = safe_float(long_call.get("strike"))
                reasons = []
                if strategy == "Iron Condor":
                    if (
                        long_put_strike is not None
                        and math.isfinite(long_put_strike)
                        and short_put_strike is not None
                        and math.isfinite(short_put_strike)
                        and short_call_strike is not None
                        and math.isfinite(short_call_strike)
                        and long_call_strike is not None
                        and math.isfinite(long_call_strike)
                        and not (long_put_strike < short_put_strike < short_call_strike < long_call_strike)
                    ):
                        reasons.append(
                            "invalid wing/short ordering (need long_put < short_put < short_call < long_call)"
                        )
                else:
                    if (
                        long_put_strike is not None
                        and math.isfinite(long_put_strike)
                        and short_put_strike is not None
                        and math.isfinite(short_put_strike)
                        and short_call_strike is not None
                        and math.isfinite(short_call_strike)
                        and long_call_strike is not None
                        and math.isfinite(long_call_strike)
                        and not (
                            long_put_strike < short_put_strike
                            and short_put_strike == short_call_strike
                            and short_call_strike < long_call_strike
                        )
                    ):
                        reasons.append(
                            "invalid fly ordering (need long_put < short_put == short_call < long_call)"
                        )
                if strategy == "Iron Condor":
                    if (
                        short_put_strike is not None
                        and math.isfinite(short_put_strike)
                        and short_call_strike is not None
                        and math.isfinite(short_call_strike)
                        and not (short_put_strike < spot_eval_live < short_call_strike)
                    ):
                        reasons.append(
                            f"spot {spot_eval_live:.2f} outside short strikes [{short_put_strike:.2f}, {short_call_strike:.2f}]"
                        )
                elif strategy == "Long Iron Condor":
                    if (
                        long_put_strike is None
                        or not math.isfinite(long_put_strike)
                        or short_put_strike is None
                        or not math.isfinite(short_put_strike)
                        or long_call_strike is None
                        or not math.isfinite(long_call_strike)
                        or short_call_strike is None
                        or not math.isfinite(short_call_strike)
                        or not (short_put_strike < long_put_strike < long_call_strike < short_call_strike)
                    ):
                        reasons.append(
                            "invalid long-condor ordering (need short_put < long_put < long_call < short_call)"
                        )
                    elif not (long_put_strike < spot_eval_live < long_call_strike):
                        reasons.append(
                            f"spot {spot_eval_live:.2f} outside long-body [{long_put_strike:.2f}, {long_call_strike:.2f}]"
                        )
                if live_net_bid_ask is not None and math.isfinite(live_net_bid_ask):
                    if strategy == "Long Iron Condor":
                        if (
                            long_put_strike is not None
                            and math.isfinite(long_put_strike)
                            and long_call_strike is not None
                            and math.isfinite(long_call_strike)
                        ):
                            be_low = long_put_strike - live_net_bid_ask
                            be_high = long_call_strike + live_net_bid_ask
                            if not (be_low < spot_eval_live < be_high):
                                reasons.append(
                                    f"spot {spot_eval_live:.2f} outside breakevens [{be_low:.2f}, {be_high:.2f}]"
                                )
                    elif (
                        short_put_strike is not None
                        and math.isfinite(short_put_strike)
                        and short_call_strike is not None
                        and math.isfinite(short_call_strike)
                    ):
                        be_low = short_put_strike - live_net_bid_ask
                        be_high = short_call_strike + live_net_bid_ask
                        if not (be_low < spot_eval_live < be_high):
                            reasons.append(
                                f"spot {spot_eval_live:.2f} outside breakevens [{be_low:.2f}, {be_high:.2f}]"
                            )
                if reasons:
                    entry_structure_ok = False
                    entry_structure_reason = "; ".join(reasons)

        live_status = "ok_live"
        if chain_status == "ERROR":
            live_status = "chain_error"
        elif chain_status != "SUCCESS":
            live_status = "chain_not_success"
        elif strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"} and (
            (not short_live_symbol)
            or (not long_live_symbol)
            or (not short_call_live_symbol)
            or (not long_call_live_symbol)
        ):
            live_status = "bad_occ_symbol"
        elif strategy not in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"} and ((not short_live_symbol) or (not long_live_symbol)):
            live_status = "bad_occ_symbol"
        elif strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"} and (
            short_in_chain is False
            or long_in_chain is False
            or short_call_in_chain is False
            or long_call_in_chain is False
        ):
            live_status = "missing_leg_in_live_chain"
        elif strategy not in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"} and (short_in_chain is False or long_in_chain is False):
            live_status = "missing_leg_in_live_chain"
        elif spot_eval_live is None:
            live_status = "missing_underlying_quote"
        elif not entry_structure_ok:
            live_status = "invalid_entry_structure"
        elif args.hard_invalidation and invalidation_breached_live is True:
            live_status = "invalidation_breached_live"
        elif live_net_bid_ask is None:
            live_status = "missing_live_quote"
        elif gate_pass_live is False:
            live_status = "fails_live_entry_gate"

        live_max_profit = None
        live_max_loss = None
        if live_net_bid_ask is not None and width_for_risk is not None:
            if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                put_width = (
                    abs(short_put["strike"] - long_put["strike"])
                    if (short_put.get("strike") is not None and long_put.get("strike") is not None)
                    else None
                )
                call_width = (
                    abs(long_call["strike"] - short_call["strike"])
                    if (short_call.get("strike") is not None and long_call.get("strike") is not None)
                    else None
                )
                if put_width is not None and call_width is not None:
                    if strategy == "Long Iron Condor":
                        live_max_profit = max(put_width - live_net_bid_ask, call_width - live_net_bid_ask) * 100.0
                        live_max_loss = live_net_bid_ask * 100.0
                    else:
                        live_max_profit = live_net_bid_ask * 100.0
                        live_max_loss = max(put_width - live_net_bid_ask, call_width - live_net_bid_ask) * 100.0
            elif net_type == "credit":
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
                "short_call_leg_live_symbol": short_call_live_symbol,
                "long_call_leg_live_symbol": long_call_live_symbol,
                "short_leg_in_live_chain": short_in_chain,
                "long_leg_in_live_chain": long_in_chain,
                "short_call_leg_in_live_chain": short_call_in_chain,
                "long_call_leg_in_live_chain": long_call_in_chain,
                "short_bid_live": short_put.get("bid"),
                "short_ask_live": short_put.get("ask"),
                "short_last_live": short_put.get("last"),
                "short_mark_live": short_put.get("mark"),
                "short_delta_live": short_put.get("delta"),
                "short_oi_live": short_put.get("oi"),
                "short_volume_live": short_put.get("volume"),
                "short_strike_live": short_put.get("strike"),
                "short_expiry_live": short_put.get("expiry"),
                "long_bid_live": long_put.get("bid"),
                "long_ask_live": long_put.get("ask"),
                "long_last_live": long_put.get("last"),
                "long_mark_live": long_put.get("mark"),
                "long_delta_live": long_put.get("delta"),
                "long_oi_live": long_put.get("oi"),
                "long_volume_live": long_put.get("volume"),
                "long_strike_live": long_put.get("strike"),
                "long_expiry_live": long_put.get("expiry"),
                "short_put_bid_live": short_put.get("bid"),
                "short_put_ask_live": short_put.get("ask"),
                "long_put_bid_live": long_put.get("bid"),
                "long_put_ask_live": long_put.get("ask"),
                "short_call_bid_live": short_call.get("bid"),
                "short_call_ask_live": short_call.get("ask"),
                "long_call_bid_live": long_call.get("bid"),
                "long_call_ask_live": long_call.get("ask"),
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
                "entry_structure_ok_live": entry_structure_ok,
                "entry_structure_reason_live": entry_structure_reason,
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

    if snapshot_out_json is not None:
        snapshot_out_json.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload = {
            "generated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "auth_mode": service.auth_mode,
            "shortlist_csv": str(shortlist_csv),
            "out_csv": str(out_csv),
            "out_final_csv": str(out_final_csv),
            "rows_shortlist_input": int(len(df)),
            "rows_enriched": int(len(live_df)),
            "rows_final_live_valid": int(len(final_df)),
            "tickers": sorted(ticker_ranges.keys()),
            "chain_status_by_ticker": chain_status_by_ticker,
            "chain_query_symbol_by_ticker": chain_query_symbol_by_ticker,
            "chain_error_by_ticker": chain_error_by_ticker,
            "underlying_query_symbols": underlying_query_symbols,
            "chain_save_dir": str(Path(args.save_chain_dir).expanduser().resolve()) if args.save_chain_dir else "",
            "hard_invalidation": bool(args.hard_invalidation),
        }
        snapshot_out_json.write_text(
            json.dumps(snapshot_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"Auth mode: {service.auth_mode}")
    print(f"Shortlist input rows: {len(df):,}")
    print(f"Enriched rows: {len(live_df):,}")
    print(f"Final live-valid rows: {len(final_df):,}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_final_csv}")
    if snapshot_out_json is not None:
        print(f"Wrote: {snapshot_out_json}")


if __name__ == "__main__":
    main()
