#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_money(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def fmt_signed(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:+.1f}"


def md_cell(value: Any) -> str:
    text = str(value or "")
    return text.replace("|", "\\|").replace("\n", " ").strip()


@dataclass(frozen=True)
class Contract:
    right: str
    expiry: str
    dte: int
    strike: float
    symbol: str
    bid: float
    ask: float
    mark: float
    delta: float
    oi: float
    volume: float
    iv: float


@dataclass
class Setup:
    rank: float
    ticker: str
    bias: str
    sentiment_score: float
    confidence: float
    spot: float
    strategy: str
    expiry: str
    dte: int
    legs: str
    long_leg: str
    short_leg: str
    net_type: str
    entry_gate: str
    entry_band: str
    natural: float
    mid: float
    width: float
    max_risk: float
    max_profit: float
    reward_risk: float
    invalidation: str
    trigger: str
    liquidity_score: float
    uw_stock_flow_score: float
    uw_options_flow_score: float
    drivers: str
    source_status: str = "OK"

    def to_row(self) -> Dict[str, Any]:
        return {
            "rank": round(self.rank, 2),
            "ticker": self.ticker,
            "bias": self.bias,
            "sentiment_score": round(self.sentiment_score, 2),
            "confidence": round(self.confidence, 2),
            "spot": round(self.spot, 2),
            "strategy": self.strategy,
            "expiry": self.expiry,
            "dte": self.dte,
            "legs": self.legs,
            "short_leg": self.short_leg,
            "long_leg": self.long_leg,
            "net_type": self.net_type,
            "entry_gate": self.entry_gate,
            "entry_band": self.entry_band,
            "natural": round(self.natural, 2),
            "mid": round(self.mid, 2),
            "width": round(self.width, 2),
            "max_risk": round(self.max_risk, 2),
            "max_profit": round(self.max_profit, 2),
            "reward_risk": round(self.reward_risk, 2),
            "invalidation": self.invalidation,
            "trigger": self.trigger,
            "liquidity_score": round(self.liquidity_score, 2),
            "uw_stock_flow_score": round(self.uw_stock_flow_score, 2),
            "uw_options_flow_score": round(self.uw_options_flow_score, 2),
            "drivers": self.drivers,
            "source_status": self.source_status,
        }


FIELDNAMES = [
    "rank",
    "ticker",
    "bias",
    "sentiment_score",
    "confidence",
    "spot",
    "strategy",
    "expiry",
    "dte",
    "legs",
    "short_leg",
    "long_leg",
    "net_type",
    "entry_gate",
    "entry_band",
    "natural",
    "mid",
    "width",
    "max_risk",
    "max_profit",
    "reward_risk",
    "invalidation",
    "trigger",
    "liquidity_score",
    "uw_stock_flow_score",
    "uw_options_flow_score",
    "drivers",
    "source_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ranked option trade cards from sentiment scores and Schwab chain JSON."
    )
    parser.add_argument("--sentiment-csv", required=True, help="Sentiment score CSV.")
    parser.add_argument(
        "--schwab-json-dir",
        required=True,
        help="Directory written by python -m uwos.schwab_quotes --save-json-dir.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--as-of", default=dt.date.today().isoformat(), help="YYYY-MM-DD report date.")
    parser.add_argument("--top-tickers", type=int, default=20, help="Sentiment tickers to inspect.")
    parser.add_argument("--max-per-ticker", type=int, default=2, help="Max cards retained per ticker.")
    parser.add_argument("--top-cards", type=int, default=12, help="Top cards to render in markdown.")
    parser.add_argument("--min-abs-score", type=float, default=8.0, help="Minimum absolute sentiment score.")
    parser.add_argument("--min-dte", type=int, default=15, help="Minimum option DTE.")
    parser.add_argument("--max-dte", type=int, default=50, help="Maximum option DTE.")
    parser.add_argument(
        "--invalid-buffer-pct",
        type=float,
        default=0.03,
        help="Directional debit invalidation buffer from spot.",
    )
    return parser.parse_args()


def read_sentiment_rows(path: Path, top_tickers: int, min_abs_score: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = safe_float(row.get("sentiment_score"))
            confidence = safe_float(row.get("confidence"))
            if not math.isfinite(score) or abs(score) < min_abs_score:
                continue
            if not math.isfinite(confidence):
                confidence = 0.0
            row["_sort_score"] = abs(score) * (0.5 + confidence / 200.0)
            rows.append(row)
    rows.sort(key=lambda r: safe_float(r.get("_sort_score")), reverse=True)
    return rows[: max(1, top_tickers)]


def midpoint(bid: float, ask: float) -> float:
    if math.isfinite(bid) and math.isfinite(ask):
        return (bid + ask) / 2.0
    return math.nan


def contract_mark(contract: Dict[str, Any]) -> float:
    mark = safe_float(contract.get("mark"))
    if math.isfinite(mark):
        return mark
    return midpoint(safe_float(contract.get("bid")), safe_float(contract.get("ask")))


def contract_rows(chain: Dict[str, Any], right: str, min_dte: int, max_dte: int) -> List[Contract]:
    map_name = "callExpDateMap" if right == "C" else "putExpDateMap"
    rows: List[Contract] = []
    for exp_key, strike_map in (chain.get(map_name) or {}).items():
        expiry = str(exp_key).split(":")[0]
        try:
            exp_dte = int(str(exp_key).split(":")[1])
        except (IndexError, ValueError):
            exp_dte = 0
        for strike_key, contracts in (strike_map or {}).items():
            strike = safe_float(strike_key)
            if not math.isfinite(strike):
                continue
            for payload in contracts or []:
                dte = int(safe_float(payload.get("daysToExpiration")) or exp_dte)
                bid = safe_float(payload.get("bid"))
                ask = safe_float(payload.get("ask"))
                mark = contract_mark(payload)
                symbol = str(payload.get("symbol") or "").strip()
                if dte < min_dte or dte > max_dte:
                    continue
                if not symbol or not (math.isfinite(bid) and math.isfinite(ask) and ask > 0.0):
                    continue
                rows.append(
                    Contract(
                        right=right,
                        expiry=expiry,
                        dte=dte,
                        strike=float(strike),
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        mark=mark,
                        delta=safe_float(payload.get("delta")),
                        oi=safe_float(payload.get("openInterest")),
                        volume=safe_float(payload.get("totalVolume")),
                        iv=safe_float(payload.get("volatility")),
                    )
                )
    return rows


def quote_spot(chain: Dict[str, Any], quotes: Dict[str, Any], ticker: str) -> float:
    spot = safe_float(chain.get("underlyingPrice"))
    if math.isfinite(spot):
        return spot
    quote = (quotes.get(ticker) or {}).get("quote", quotes.get(ticker) or {})
    for key in ("lastPrice", "mark", "bidPrice", "askPrice"):
        spot = safe_float(quote.get(key))
        if math.isfinite(spot):
            return spot
    return math.nan


def leg_liquidity_score(contract: Contract) -> float:
    mid = midpoint(contract.bid, contract.ask)
    if not math.isfinite(mid) or mid <= 0:
        return -50.0
    width_pct = (contract.ask - contract.bid) / max(0.05, mid)
    oi = 0.0 if not math.isfinite(contract.oi) else max(0.0, contract.oi)
    volume = 0.0 if not math.isfinite(contract.volume) else max(0.0, contract.volume)
    activity = math.log1p(oi + volume) * 5.0
    return min(30.0, activity) - width_pct * 20.0


def has_delta(contract: Contract, lo: float, hi: float) -> bool:
    return not math.isfinite(contract.delta) or lo <= contract.delta <= hi


def bounded(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(hi, max(lo, value))


def setup_rank(
    row: Dict[str, Any],
    net_type: str,
    natural: float,
    width: float,
    long_leg: Contract,
    short_leg: Contract,
) -> Tuple[float, float, float]:
    sentiment_score = abs(safe_float(row.get("sentiment_score")))
    confidence = safe_float(row.get("confidence"))
    stock_flow = safe_float(row.get("uw_stock_flow_score"))
    options_flow = safe_float(row.get("options_flow_score"))
    liquidity = leg_liquidity_score(long_leg) + leg_liquidity_score(short_leg)

    if net_type == "debit":
        ratio = natural / width if width > 0 else 1.0
        reward_risk = (width - natural) / natural if natural > 0 else 0.0
        economics = 30.0 * (1.0 - ratio) + min(20.0, reward_risk * 8.0)
    else:
        ratio = natural / width if width > 0 else 0.0
        reward_risk = natural / (width - natural) if width > natural else 0.0
        economics = 55.0 * ratio + min(15.0, reward_risk * 40.0)

    flow_bonus = bounded(options_flow, -10.0, 15.0) + bounded(stock_flow, -8.0, 8.0) * 0.35
    direction_bonus = 4.0 if str(row.get("direction", "")).lower() in {"bullish", "bearish"} else 0.0
    rank = sentiment_score * 1.35 + confidence * 0.25 + economics + liquidity * 0.45 + flow_bonus + direction_bonus
    return rank, liquidity, reward_risk


def add_setup(
    out: List[Setup],
    row: Dict[str, Any],
    ticker: str,
    bias: str,
    spot: float,
    strategy: str,
    net_type: str,
    long_leg: Contract,
    short_leg: Contract,
    natural: float,
    mid: float,
    width: float,
    invalidation: str,
    trigger: str,
) -> None:
    if not (math.isfinite(width) and math.isfinite(natural) and math.isfinite(mid)):
        return
    if width <= 0.0 or natural <= 0.0:
        return
    if net_type == "debit":
        if natural >= width or natural / width > 0.72:
            return
        max_risk = natural
        max_profit = width - natural
        entry_floor = max(0.01, min(mid, natural))
        entry_band = f"{entry_floor:.2f}-{natural:.2f} debit"
        entry_gate = f"<= {natural:.2f} db"
    else:
        if natural / width < 0.08 or natural >= width:
            return
        max_risk = width - natural
        max_profit = natural
        entry_band = f">= {natural:.2f} credit; aim {max(natural, mid):.2f}"
        entry_gate = f">= {natural:.2f} cr"

    rank, liquidity, reward_risk = setup_rank(row, net_type, natural, width, long_leg, short_leg)
    score = safe_float(row.get("sentiment_score"))
    confidence = safe_float(row.get("confidence"))
    out.append(
        Setup(
            rank=rank,
            ticker=ticker,
            bias=bias,
            sentiment_score=score,
            confidence=confidence,
            spot=spot,
            strategy=strategy,
            expiry=long_leg.expiry,
            dte=long_leg.dte,
            legs=f"Buy {long_leg.strike:g}{long_leg.right} / Sell {short_leg.strike:g}{short_leg.right}",
            long_leg=long_leg.symbol,
            short_leg=short_leg.symbol,
            net_type=net_type,
            entry_gate=entry_gate,
            entry_band=entry_band,
            natural=natural,
            mid=mid,
            width=width,
            max_risk=max_risk,
            max_profit=max_profit,
            reward_risk=reward_risk,
            invalidation=invalidation,
            trigger=trigger,
            liquidity_score=liquidity,
            uw_stock_flow_score=safe_float(row.get("uw_stock_flow_score")),
            uw_options_flow_score=safe_float(row.get("options_flow_score")),
            drivers=str(row.get("drivers") or ""),
        )
    )


def grouped_by_expiry(rows: Sequence[Contract]) -> List[Tuple[str, List[Contract]]]:
    grouped: Dict[str, List[Contract]] = {}
    for row in rows:
        grouped.setdefault(row.expiry, []).append(row)
    return sorted(
        grouped.items(),
        key=lambda item: abs((item[1][0].dte if item[1] else 30) - 30),
    )


def bullish_setups(
    out: List[Setup],
    row: Dict[str, Any],
    ticker: str,
    spot: float,
    calls: Sequence[Contract],
    puts: Sequence[Contract],
    invalid_buffer_pct: float,
) -> None:
    invalid_debit = spot * (1.0 - invalid_buffer_pct)
    for expiry, exp_calls in grouped_by_expiry(calls)[:4]:
        exp_calls = sorted(exp_calls, key=lambda c: c.strike)
        longs = [
            c
            for c in exp_calls
            if spot * 0.96 <= c.strike <= spot * 1.04 and has_delta(c, 0.38, 0.62)
        ]
        for long_leg in longs:
            for short_leg in exp_calls:
                if short_leg.strike <= long_leg.strike or short_leg.strike > spot * 1.15:
                    continue
                width = short_leg.strike - long_leg.strike
                if width < max(1.0, spot * 0.015) or width > max(30.0, spot * 0.15):
                    continue
                natural = long_leg.ask - short_leg.bid
                mid = long_leg.mark - short_leg.mark
                trigger = (
                    f"underlying reclaim {long_leg.strike:.2f}"
                    if long_leg.strike > spot * 1.005
                    else f"underlying holds above {spot * 0.995:.2f}"
                )
                add_setup(
                    out,
                    row,
                    ticker,
                    "bullish",
                    spot,
                    "Bull Call Debit",
                    "debit",
                    long_leg,
                    short_leg,
                    round(natural, 2),
                    round(mid, 2),
                    round(width, 2),
                    f"close < {invalid_debit:.2f}",
                    trigger,
                )

    put_by_exp = {expiry: rows for expiry, rows in grouped_by_expiry(puts)}
    for expiry, exp_puts in list(put_by_exp.items())[:4]:
        exp_puts = sorted(exp_puts, key=lambda c: c.strike)
        shorts = [
            p
            for p in exp_puts
            if p.strike < spot and has_delta(p, -0.38, -0.16)
        ]
        for short_leg in shorts:
            for long_leg in exp_puts:
                if long_leg.strike >= short_leg.strike:
                    continue
                width = short_leg.strike - long_leg.strike
                if width < max(1.0, spot * 0.015) or width > max(30.0, spot * 0.15):
                    continue
                natural = short_leg.bid - long_leg.ask
                mid = short_leg.mark - long_leg.mark
                add_setup(
                    out,
                    row,
                    ticker,
                    "bullish",
                    spot,
                    "Bull Put Credit",
                    "credit",
                    long_leg,
                    short_leg,
                    round(natural, 2),
                    round(mid, 2),
                    round(width, 2),
                    f"close < {short_leg.strike:.2f}",
                    f"underlying holds above {short_leg.strike:.2f}",
                )


def bearish_setups(
    out: List[Setup],
    row: Dict[str, Any],
    ticker: str,
    spot: float,
    calls: Sequence[Contract],
    puts: Sequence[Contract],
    invalid_buffer_pct: float,
) -> None:
    invalid_debit = spot * (1.0 + invalid_buffer_pct)
    for expiry, exp_puts in grouped_by_expiry(puts)[:4]:
        exp_puts = sorted(exp_puts, key=lambda c: c.strike)
        longs = [
            p
            for p in exp_puts
            if spot * 0.96 <= p.strike <= spot * 1.04 and has_delta(p, -0.62, -0.38)
        ]
        for long_leg in longs:
            for short_leg in exp_puts:
                if short_leg.strike >= long_leg.strike or short_leg.strike < spot * 0.85:
                    continue
                width = long_leg.strike - short_leg.strike
                if width < max(1.0, spot * 0.015) or width > max(30.0, spot * 0.15):
                    continue
                natural = long_leg.ask - short_leg.bid
                mid = long_leg.mark - short_leg.mark
                trigger = (
                    f"underlying lose {long_leg.strike:.2f}"
                    if long_leg.strike < spot * 0.995
                    else f"underlying stays below {spot * 1.005:.2f}"
                )
                add_setup(
                    out,
                    row,
                    ticker,
                    "bearish",
                    spot,
                    "Bear Put Debit",
                    "debit",
                    long_leg,
                    short_leg,
                    round(natural, 2),
                    round(mid, 2),
                    round(width, 2),
                    f"close > {invalid_debit:.2f}",
                    trigger,
                )

    for expiry, exp_calls in grouped_by_expiry(calls)[:4]:
        exp_calls = sorted(exp_calls, key=lambda c: c.strike)
        shorts = [
            c
            for c in exp_calls
            if c.strike > spot and has_delta(c, 0.16, 0.38)
        ]
        for short_leg in shorts:
            for long_leg in exp_calls:
                if long_leg.strike <= short_leg.strike:
                    continue
                width = long_leg.strike - short_leg.strike
                if width < max(1.0, spot * 0.015) or width > max(30.0, spot * 0.15):
                    continue
                natural = short_leg.bid - long_leg.ask
                mid = short_leg.mark - long_leg.mark
                add_setup(
                    out,
                    row,
                    ticker,
                    "bearish",
                    spot,
                    "Bear Call Credit",
                    "credit",
                    long_leg,
                    short_leg,
                    round(natural, 2),
                    round(mid, 2),
                    round(width, 2),
                    f"close > {short_leg.strike:.2f}",
                    f"underlying stays below {short_leg.strike:.2f}",
                )


def build_setups(
    sentiment_rows: Sequence[Dict[str, Any]],
    schwab_dir: Path,
    min_dte: int,
    max_dte: int,
    invalid_buffer_pct: float,
) -> Tuple[List[Setup], List[Dict[str, Any]]]:
    quotes_path = schwab_dir / "quotes.json"
    quotes = json.loads(quotes_path.read_text(encoding="utf-8")) if quotes_path.is_file() else {}
    setups: List[Setup] = []
    rejects: List[Dict[str, Any]] = []
    for row in sentiment_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        chain_path = schwab_dir / f"option_chain_{ticker}.json"
        if not chain_path.is_file():
            rejects.append({"ticker": ticker, "reason": "missing_schwab_chain"})
            continue
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        if str(chain.get("status", "")).upper() != "SUCCESS":
            rejects.append({"ticker": ticker, "reason": f"chain_status_{chain.get('status', 'UNKNOWN')}"})
            continue
        spot = quote_spot(chain, quotes, ticker)
        if not math.isfinite(spot) or spot <= 0.0:
            rejects.append({"ticker": ticker, "reason": "missing_spot"})
            continue
        calls = contract_rows(chain, "C", min_dte, max_dte)
        puts = contract_rows(chain, "P", min_dte, max_dte)
        before = len(setups)
        score = safe_float(row.get("sentiment_score"))
        if score >= 0:
            bullish_setups(setups, row, ticker, spot, calls, puts, invalid_buffer_pct)
        else:
            bearish_setups(setups, row, ticker, spot, calls, puts, invalid_buffer_pct)
        if len(setups) == before:
            rejects.append({"ticker": ticker, "reason": "no_viable_spread_after_liquidity_and_economics"})
    return setups, rejects


def dedupe_setups(setups: Sequence[Setup], max_per_ticker: int) -> List[Setup]:
    best_by_strategy: Dict[Tuple[str, str], Setup] = {}
    for setup in setups:
        key = (setup.ticker, setup.strategy)
        if key not in best_by_strategy or setup.rank > best_by_strategy[key].rank:
            best_by_strategy[key] = setup

    out: List[Setup] = []
    by_ticker: Dict[str, List[Setup]] = {}
    for setup in best_by_strategy.values():
        by_ticker.setdefault(setup.ticker, []).append(setup)
    for ticker_setups in by_ticker.values():
        out.extend(sorted(ticker_setups, key=lambda s: s.rank, reverse=True)[: max(1, max_per_ticker)])
    return sorted(out, key=lambda s: s.rank, reverse=True)


def write_csv(path: Path, setups: Sequence[Setup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for setup in setups:
            writer.writerow(setup.to_row())


def write_rejects(path: Path, rejects: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "reason"])
        writer.writeheader()
        for row in rejects:
            writer.writerow(row)


def render_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(md_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(c) for c in row) + " |")
    return "\n".join(lines)


def concise_driver(text: str, max_len: int = 130) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def write_report(
    path: Path,
    setups: Sequence[Setup],
    rejects: Sequence[Dict[str, Any]],
    as_of: str,
    sentiment_csv: Path,
    schwab_dir: Path,
    csv_path: Path,
    top_cards: int,
) -> None:
    top = list(setups[: max(1, top_cards)])
    lines: List[str] = []
    lines.append(f"# Sentiment Trade Setups - {as_of}")
    lines.append("")
    lines.append("Pure sentiment handoff: sentiment scores + UW flow context + Schwab live chains. No trend-analysis artifact is used here.")
    lines.append("")
    lines.append("## Top Order Cards")
    if top:
        rows = []
        for idx, setup in enumerate(top, start=1):
            why = f"{fmt_signed(setup.sentiment_score)} sentiment, {setup.confidence:.0f} conf; {concise_driver(setup.drivers, 95)}"
            rows.append(
                [
                    idx,
                    setup.ticker,
                    setup.bias,
                    setup.strategy,
                    setup.expiry,
                    setup.legs,
                    setup.entry_band,
                    setup.invalidation,
                    setup.trigger,
                    why,
                ]
            )
        lines.append(
            render_table(
                rows,
                ["#", "Ticker", "Bias", "Strategy", "Exp", "Legs", "Entry", "Invalid", "Trigger", "Why"],
            )
        )
    else:
        lines.append("No viable spreads survived liquidity/economics filters.")
    lines.append("")
    lines.append("## Risk Math")
    if top:
        rows = []
        for setup in top:
            rows.append(
                [
                    setup.ticker,
                    f"${fmt_money(setup.spot)}",
                    f"${fmt_money(setup.width)}",
                    f"${fmt_money(setup.max_risk)}",
                    f"${fmt_money(setup.max_profit)}",
                    f"{setup.reward_risk:.2f}",
                    f"{setup.liquidity_score:.1f}",
                ]
            )
        lines.append(render_table(rows, ["Ticker", "Spot", "Width", "Max Risk", "Max Profit", "R/R", "Liq"]))
    lines.append("")
    if rejects:
        lines.append("## Rejected")
        lines.append(render_table([[r.get("ticker"), r.get("reason")] for r in rejects], ["Ticker", "Reason"]))
        lines.append("")
    lines.append("## Files")
    lines.append(f"- Sentiment CSV: `{sentiment_csv}`")
    lines.append(f"- Schwab JSON dir: `{schwab_dir}`")
    lines.append(f"- Setup CSV: `{csv_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sentiment_csv = Path(args.sentiment_csv).expanduser().resolve()
    schwab_dir = Path(args.schwab_json_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_sentiment_rows(sentiment_csv, args.top_tickers, args.min_abs_score)
    setups, rejects = build_setups(
        rows,
        schwab_dir=schwab_dir,
        min_dte=int(args.min_dte),
        max_dte=int(args.max_dte),
        invalid_buffer_pct=float(args.invalid_buffer_pct),
    )
    ranked = dedupe_setups(setups, max_per_ticker=int(args.max_per_ticker))

    csv_path = out_dir / f"sentiment-trade-setups-{args.as_of}.csv"
    reject_path = out_dir / f"sentiment-trade-setup-rejects-{args.as_of}.csv"
    md_path = out_dir / f"sentiment-trade-setups-{args.as_of}.md"
    write_csv(csv_path, ranked)
    write_rejects(reject_path, rejects)
    write_report(
        md_path,
        ranked,
        rejects,
        as_of=str(args.as_of),
        sentiment_csv=sentiment_csv,
        schwab_dir=schwab_dir,
        csv_path=csv_path,
        top_cards=int(args.top_cards),
    )
    print(f"Generated setups: {len(ranked)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if rejects:
        print(f"Rejected tickers: {len(rejects)} ({reject_path})")


if __name__ == "__main__":
    main()
