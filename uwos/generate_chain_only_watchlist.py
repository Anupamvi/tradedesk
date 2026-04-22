#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uwos.schwab_auth import (  # noqa: E402
    SchwabAuthConfig,
    SchwabLiveDataService,
    compact_occ_to_schwab_symbol,
)


COMPACT_OCC_RE = re.compile(r"^([A-Z\.]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass
class Candidate:
    ticker: str
    direction: str
    strategy: str
    expiry: dt.date
    dte: int
    spot: float
    lead_symbol: str
    lead_right: str
    lead_strike: float
    pair_symbol: str
    pair_strike: float
    target_value: float
    stretch_value: float
    width: float
    breakeven: float
    move_needed_pct: float
    flow_conviction: int
    flow_conviction_label: str
    geometry_label: str
    gross_directional_notional: float
    exact_flow_note: str
    include_reason: str


@dataclass
class Exclusion:
    ticker: str
    status: str
    reason: str
    exact_flow_note: str = ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a deterministic chain-only morning watchlist using local UW chain-OI plus live Schwab quotes."
    )
    ap.add_argument("--date", required=True, help="Trade date, e.g. 2026-04-22")
    ap.add_argument(
        "--base-dir",
        default=str(ROOT),
        help="Base tradedesk directory containing dated folders.",
    )
    ap.add_argument(
        "--chain-csv",
        default="",
        help="Optional explicit chain-oi-changes CSV path. Defaults to <base>/<date>/chain-oi-changes-<date>.csv",
    )
    ap.add_argument(
        "--output-md",
        default="",
        help="Optional markdown output path. Defaults to <base>/<date>/morning-watch-setups-<date>.md",
    )
    ap.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV output path. Defaults to <base>/<date>/morning-watch-setups-<date>.csv",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum number of included setups.",
    )
    ap.add_argument(
        "--min-dte",
        type=int,
        default=7,
        help="Minimum DTE to consider.",
    )
    ap.add_argument(
        "--max-dte",
        type=int,
        default=60,
        help="Maximum DTE to consider.",
    )
    ap.add_argument(
        "--focus-tickers",
        default="NFLX,ONDS,ASTS",
        help="Comma-separated tickers to always audit even if excluded.",
    )
    return ap.parse_args()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def _parse_compact_option(symbol: str) -> Optional[Tuple[str, dt.date, str, float]]:
    text = str(symbol or "").strip().upper()
    m = COMPACT_OCC_RE.match(text)
    if not m:
        return None
    root, yymmdd, right, strike8 = m.groups()
    expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike8) / 1000.0
    return root, expiry, right, strike


def _load_chain(path: Path, min_dte: int, max_dte: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    parsed = df["option_symbol"].map(_parse_compact_option)
    df = df[parsed.notna()].copy()
    df["parsed"] = parsed
    df["ticker"] = df["parsed"].map(lambda x: x[0])
    df["expiry"] = df["parsed"].map(lambda x: x[1])
    df["right"] = df["parsed"].map(lambda x: x[2])
    df["parsed_strike"] = df["parsed"].map(lambda x: x[3])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(df["parsed_strike"])
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
    df = df[df["dte"].between(min_dte, max_dte, inclusive="both")].copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0.0)
    df["prev_ask_volume"] = pd.to_numeric(df["prev_ask_volume"], errors="coerce").fillna(0.0)
    df["prev_bid_volume"] = pd.to_numeric(df["prev_bid_volume"], errors="coerce").fillna(0.0)
    df["oi_diff_plain"] = pd.to_numeric(df["oi_diff_plain"], errors="coerce").fillna(0.0)
    df["stock_price"] = pd.to_numeric(df.get("stock_price"), errors="coerce")
    df["volume_eff"] = df["volume"].clip(lower=1.0)
    df["notional"] = df["avg_price"].abs() * df["volume_eff"] * 100.0
    df["ask_share"] = (df["prev_ask_volume"] / df["volume_eff"]).clip(lower=0.0, upper=1.0)
    df["bid_share"] = (df["prev_bid_volume"] / df["volume_eff"]).clip(lower=0.0, upper=1.0)
    df["oi_factor"] = (1.0 + (df["oi_diff_plain"].abs() / df["volume_eff"]).clip(lower=0.0, upper=1.5)).fillna(1.0)
    is_call = df["right"] == "C"
    is_put = df["right"] == "P"
    df["bullish_score"] = 0.0
    df["bearish_score"] = 0.0
    df.loc[is_call, "bullish_score"] = df.loc[is_call, "notional"] * df.loc[is_call, "ask_share"] * df.loc[is_call, "oi_factor"]
    df.loc[is_call, "bearish_score"] = df.loc[is_call, "notional"] * df.loc[is_call, "bid_share"] * df.loc[is_call, "oi_factor"]
    df.loc[is_put, "bullish_score"] = df.loc[is_put, "notional"] * df.loc[is_put, "bid_share"] * df.loc[is_put, "oi_factor"]
    df.loc[is_put, "bearish_score"] = df.loc[is_put, "notional"] * df.loc[is_put, "ask_share"] * df.loc[is_put, "oi_factor"]
    df["row_direction"] = df.apply(
        lambda row: "bullish" if row["bullish_score"] >= row["bearish_score"] else "bearish",
        axis=1,
    )
    df["row_edge"] = (df[["bullish_score", "bearish_score"]].max(axis=1) - df[["bullish_score", "bearish_score"]].min(axis=1)).fillna(0.0)
    return df


def _choose_width(spot: float) -> float:
    if spot < 20:
        return 2.5
    if spot < 75:
        return 5.0
    if spot < 200:
        return 10.0
    return 20.0


def _conviction_label(score: int) -> str:
    if score >= 72:
        return "🟩 High"
    if score >= 58:
        return "🟨 Medium-High"
    if score >= 45:
        return "🟧 Medium"
    return "🟥 Low"


def _geometry_label(move_needed_pct: float) -> str:
    if move_needed_pct <= 2.5:
        return "🟩 Easier"
    if move_needed_pct <= 5.0:
        return "🟨 Moderate"
    if move_needed_pct <= 8.0:
        return "🟧 Harder"
    return "🟥 Lower"


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _quote_field(payload: Dict[str, Any], field: str) -> Optional[float]:
    body = payload.get("quote", payload)
    return _safe_float(body.get(field))


def _underlying_last(payload: Dict[str, Any]) -> Optional[float]:
    body = payload.get("quote", payload)
    return _safe_float(body.get("lastPrice")) or _safe_float(body.get("mark"))


def _build_flow_note(row: pd.Series, direction: str) -> str:
    ask = int(_safe_float(row.get("prev_ask_volume"), 0) or 0)
    bid = int(_safe_float(row.get("prev_bid_volume"), 0) or 0)
    oi = int(_safe_float(row.get("oi_diff_plain"), 0) or 0)
    right = str(row["right"])
    strike = _safe_float(row["strike"], 0.0) or 0.0
    expiry = row["expiry"].isoformat()
    if direction == "bullish" and right == "C":
        signal = "ask-side call buying"
    elif direction == "bullish" and right == "P":
        signal = "bid-side put selling"
    elif direction == "bearish" and right == "P":
        signal = "ask-side put buying"
    else:
        signal = "bid-side call selling"
    return f"{expiry} {strike:g}{right} {signal}; ask={ask}, bid={bid}, oi_change={oi}"


def _pick_pair(rows: pd.DataFrame, lead_row: pd.Series, strategy: str, spot: float) -> Optional[pd.Series]:
    target_width = _choose_width(spot)
    same_expiry = rows[(rows["expiry"] == lead_row["expiry"]) & (rows["right"] == lead_row["right"])].copy()
    if same_expiry.empty:
        return None
    lead_strike = float(lead_row["strike"])
    if strategy == "Bull Call Debit":
        same_expiry = same_expiry[same_expiry["strike"] > lead_strike]
        ideal = lead_strike + target_width
    elif strategy == "Bear Put Debit":
        same_expiry = same_expiry[same_expiry["strike"] < lead_strike]
        ideal = lead_strike - target_width
    elif strategy == "Bear Call Credit":
        same_expiry = same_expiry[same_expiry["strike"] > lead_strike]
        ideal = lead_strike + target_width
    elif strategy == "Bull Put Credit":
        same_expiry = same_expiry[same_expiry["strike"] < lead_strike]
        ideal = lead_strike - target_width
    else:
        return None
    if same_expiry.empty:
        return None
    same_expiry["strike_distance"] = (same_expiry["strike"] - ideal).abs()
    same_expiry["liq_rank"] = same_expiry["notional"].rank(ascending=False, method="dense")
    same_expiry = same_expiry.sort_values(["strike_distance", "liq_rank", "row_edge"], ascending=[True, True, False])
    return same_expiry.iloc[0]


def _quote_setup(
    svc: SchwabLiveDataService,
    ticker: str,
    strategy: str,
    lead_row: pd.Series,
    pair_row: pd.Series,
    fallback_spot: float,
    conviction_score: int,
    gross_directional_notional: float,
    include_reason: str,
) -> Optional[Candidate]:
    lead_sym = compact_occ_to_schwab_symbol(str(lead_row["option_symbol"]))
    pair_sym = compact_occ_to_schwab_symbol(str(pair_row["option_symbol"]))
    quotes = svc.get_quotes([ticker, lead_sym, pair_sym])
    stock_payload = quotes.get(ticker)
    lead_payload = quotes.get(lead_sym)
    pair_payload = quotes.get(pair_sym)
    if not stock_payload or not lead_payload or not pair_payload:
        return None
    spot = _underlying_last(stock_payload) or fallback_spot
    if not spot:
        return None
    lead_bid = _quote_field(lead_payload, "bidPrice")
    lead_ask = _quote_field(lead_payload, "askPrice")
    pair_bid = _quote_field(pair_payload, "bidPrice")
    pair_ask = _quote_field(pair_payload, "askPrice")
    if None in (lead_bid, lead_ask, pair_bid, pair_ask):
        return None
    lead_mid = _mid(lead_bid, lead_ask)
    pair_mid = _mid(pair_bid, pair_ask)
    if lead_mid is None or pair_mid is None:
        return None

    lead_strike = float(lead_row["strike"])
    pair_strike = float(pair_row["strike"])
    width = abs(pair_strike - lead_strike)

    if strategy in {"Bull Call Debit", "Bear Put Debit"}:
        target_value = round(max(0.0, lead_mid - pair_mid), 2)
        stretch_value = round(max(0.0, lead_ask - pair_bid), 2)
        if target_value <= 0:
            return None
        if strategy == "Bull Call Debit":
            breakeven = lead_strike + target_value
            move_needed_pct = max(0.0, (breakeven - spot) / spot * 100.0)
            direction = "bullish"
        else:
            breakeven = lead_strike - target_value
            move_needed_pct = max(0.0, (spot - breakeven) / spot * 100.0)
            direction = "bearish"
    else:
        target_value = round(max(0.0, lead_mid - pair_mid), 2)
        stretch_value = round(max(0.0, lead_bid - pair_ask), 2)
        if target_value <= 0:
            return None
        if strategy == "Bear Call Credit":
            breakeven = lead_strike + target_value
            move_needed_pct = max(0.0, (breakeven - spot) / spot * 100.0)
            direction = "bearish"
        else:
            breakeven = lead_strike - target_value
            move_needed_pct = max(0.0, (spot - breakeven) / spot * 100.0)
            direction = "bullish"

    return Candidate(
        ticker=ticker,
        direction=direction,
        strategy=strategy,
        expiry=lead_row["expiry"],
        dte=int(_safe_float(lead_row["dte"], 0) or 0),
        spot=round(float(spot), 4),
        lead_symbol=lead_sym,
        lead_right=str(lead_row["right"]),
        lead_strike=lead_strike,
        pair_symbol=pair_sym,
        pair_strike=pair_strike,
        target_value=target_value,
        stretch_value=stretch_value,
        width=width,
        breakeven=round(breakeven, 2),
        move_needed_pct=round(move_needed_pct, 2),
        flow_conviction=conviction_score,
        flow_conviction_label=_conviction_label(conviction_score),
        geometry_label=_geometry_label(move_needed_pct),
        gross_directional_notional=round(gross_directional_notional, 2),
        exact_flow_note=_build_flow_note(lead_row, direction),
        include_reason=include_reason,
    )


def _ticker_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    bullish = float(rows["bullish_score"].sum())
    bearish = float(rows["bearish_score"].sum())
    total = bullish + bearish
    dominant = "bullish" if bullish >= bearish else "bearish"
    dominant_score = max(bullish, bearish)
    opposing_score = min(bullish, bearish)
    conviction = int(round(100.0 * (dominant_score - opposing_score) / total)) if total > 0 else 0
    confirmation = min(1.0, float(rows["oi_diff_plain"].abs().sum()) / max(1.0, float(rows["volume_eff"].sum())))
    conviction = max(0, min(99, int(round(conviction * 0.7 + confirmation * 30.0))))
    return {
        "bullish_score": bullish,
        "bearish_score": bearish,
        "dominant": dominant,
        "dominant_score": dominant_score,
        "opposing_score": opposing_score,
        "conviction": conviction,
        "gross_directional_notional": total,
    }


def _build_candidate_for_ticker(
    svc: SchwabLiveDataService,
    ticker: str,
    rows: pd.DataFrame,
) -> Tuple[Optional[Candidate], Optional[Exclusion]]:
    summary = _ticker_summary(rows)
    spot_fallback = _safe_float(rows["stock_price"].dropna().iloc[0] if rows["stock_price"].notna().any() else None, 0.0) or 0.0
    dominant = summary["dominant"]
    conviction = int(summary["conviction"])

    calls = rows[rows["right"] == "C"].copy()
    puts = rows[rows["right"] == "P"].copy()

    if dominant == "bullish":
        bullish_calls = calls.sort_values(["bullish_score", "row_edge", "notional"], ascending=False)
        if bullish_calls.empty or float(bullish_calls.iloc[0]["bullish_score"]) <= float(bullish_calls.iloc[0]["bearish_score"]):
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="No clean bullish call-buying lead contract after side-aware ranking.",
            )
        lead = bullish_calls.iloc[0]
        pair = _pick_pair(calls, lead, "Bull Call Debit", spot_fallback)
        if pair is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Could not find a higher-strike call pair for a bull call debit.",
                exact_flow_note=_build_flow_note(lead, dominant),
            )
        candidate = _quote_setup(
            svc=svc,
            ticker=ticker,
            strategy="Bull Call Debit",
            lead_row=lead,
            pair_row=pair,
            fallback_spot=spot_fallback,
            conviction_score=conviction,
            gross_directional_notional=float(summary["gross_directional_notional"]),
            include_reason="Top bullish exact-leg flow was call ask-side buying and paired cleanly into a debit spread.",
        )
        if candidate is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Live Schwab quote coverage failed for the bullish spread legs.",
                exact_flow_note=_build_flow_note(lead, dominant),
            )
        return candidate, None

    bearish_puts = puts.sort_values(["bearish_score", "row_edge", "notional"], ascending=False)
    bearish_calls = calls.sort_values(["bearish_score", "row_edge", "notional"], ascending=False)
    lead_put = bearish_puts.iloc[0] if not bearish_puts.empty else None
    lead_call = bearish_calls.iloc[0] if not bearish_calls.empty else None
    put_score = float(lead_put["bearish_score"]) if lead_put is not None else 0.0
    call_score = float(lead_call["bearish_score"]) if lead_call is not None else 0.0

    if put_score >= call_score and lead_put is not None and put_score > float(lead_put["bullish_score"]):
        pair = _pick_pair(puts, lead_put, "Bear Put Debit", spot_fallback)
        if pair is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Could not find a lower-strike put pair for a bear put debit.",
                exact_flow_note=_build_flow_note(lead_put, dominant),
            )
        candidate = _quote_setup(
            svc=svc,
            ticker=ticker,
            strategy="Bear Put Debit",
            lead_row=lead_put,
            pair_row=pair,
            fallback_spot=spot_fallback,
            conviction_score=conviction,
            gross_directional_notional=float(summary["gross_directional_notional"]),
            include_reason="Top bearish exact-leg flow was put ask-side buying and paired cleanly into a debit spread.",
        )
        if candidate is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Live Schwab quote coverage failed for the bearish put spread legs.",
                exact_flow_note=_build_flow_note(lead_put, dominant),
            )
        return candidate, None

    if lead_call is not None and call_score > float(lead_call["bullish_score"]):
        pair = _pick_pair(calls, lead_call, "Bear Call Credit", spot_fallback)
        if pair is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Could not find a higher-strike call pair for a bear call credit.",
                exact_flow_note=_build_flow_note(lead_call, dominant),
            )
        candidate = _quote_setup(
            svc=svc,
            ticker=ticker,
            strategy="Bear Call Credit",
            lead_row=lead_call,
            pair_row=pair,
            fallback_spot=spot_fallback,
            conviction_score=conviction,
            gross_directional_notional=float(summary["gross_directional_notional"]),
            include_reason="Top bearish exact-leg flow was call bid-side selling and paired into a call credit spread.",
        )
        if candidate is None:
            return None, Exclusion(
                ticker=ticker,
                status="EXCLUDED",
                reason="Live Schwab quote coverage failed for the bearish call spread legs.",
                exact_flow_note=_build_flow_note(lead_call, dominant),
            )
        return candidate, None

    strongest_row = rows.sort_values(["row_edge", "notional"], ascending=False).iloc[0]
    return None, Exclusion(
        ticker=ticker,
        status="EXCLUDED",
        reason="Directional evidence was mixed; exact-leg side-aware flow did not support a clean spread structure.",
        exact_flow_note=_build_flow_note(strongest_row, summary["dominant"]),
    )


def _rank_candidate(candidate: Candidate) -> float:
    geometry_bonus = max(0.0, 10.0 - candidate.move_needed_pct)
    return candidate.flow_conviction * 1000.0 + geometry_bonus * 100.0 + min(candidate.gross_directional_notional, 5_000_000.0) / 100.0


def _rank_ticker_summary(summary: Dict[str, Any]) -> float:
    return (
        float(summary["conviction"]) * 1000.0
        + min(float(summary["dominant_score"]), 5_000_000.0) / 100.0
        + min(float(summary["gross_directional_notional"]), 10_000_000.0) / 1000.0
    )


def _render_markdown(
    run_date: dt.date,
    included: Sequence[Candidate],
    exclusions: Sequence[Exclusion],
    focus_audit: Sequence[Exclusion],
) -> str:
    lines: List[str] = []
    lines.append(f"# Deterministic Chain-Only Morning Watch - {run_date.isoformat()}")
    lines.append("")
    lines.append("Built from local `chain-oi-changes` plus **live Schwab quotes**. This is the coded pre-open watch layer, not the full 4-file daily pipeline.")
    lines.append("")
    lines.append("## Included setups")
    lines.append("")
    lines.append("| Ticker | Dir | Strategy | Expiry | DTE | Legs | Target | Stretch/Floor | Spot | Breakeven | Move Needed | Flow Conviction | Breakeven Difficulty | Include Reason | Exact Flow |")
    lines.append("|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for c in included:
        if "Credit" in c.strategy:
            target_label = f"{c.target_value:.2f} cr"
            stretch_label = f"{c.stretch_value:.2f} floor"
        else:
            target_label = f"{c.target_value:.2f} db"
            stretch_label = f"{c.stretch_value:.2f} max"
        lead_leg = f"{'Sell' if 'Credit' in c.strategy and c.strategy.startswith('Bear Call') else 'Buy'} {c.lead_strike:g}{c.lead_right}"
        if c.strategy == "Bull Call Debit":
            pair_leg = f"Sell {c.pair_strike:g}C"
        elif c.strategy == "Bear Put Debit":
            pair_leg = f"Sell {c.pair_strike:g}P"
        elif c.strategy == "Bear Call Credit":
            pair_leg = f"Buy {c.pair_strike:g}C"
        else:
            pair_leg = f"Buy {c.pair_strike:g}P"
        lines.append(
            f"| {c.ticker} | {'▲' if c.direction == 'bullish' else '▼'} | {c.strategy} | {c.expiry.isoformat()} | {c.dte} | {lead_leg} / {pair_leg} | {target_label} | {stretch_label} | {c.spot:.2f} | {c.breakeven:.2f} | {c.move_needed_pct:.2f}% | {c.flow_conviction_label} ({c.flow_conviction}) | {c.geometry_label} | {c.include_reason} | {c.exact_flow_note} |"
        )
    if not included:
        lines.append("_No coded chain-only watch setups cleared live quote and directional-flow checks._")
    lines.append("")
    lines.append("## Focus-ticker audit")
    lines.append("")
    lines.append("| Ticker | Status | Reason | Exact Flow |")
    lines.append("|---|---|---|---|")
    focus_map: Dict[str, Exclusion] = {x.ticker: x for x in focus_audit}
    included_map: Dict[str, Candidate] = {x.ticker: x for x in included}
    all_focus = list(dict.fromkeys([*included_map.keys(), *focus_map.keys()]))
    for ticker in all_focus:
        if ticker in included_map:
            c = included_map[ticker]
            lines.append(
                f"| {ticker} | INCLUDED | {c.include_reason} | {c.exact_flow_note} |"
            )
        else:
            ex = focus_map[ticker]
            lines.append(
                f"| {ticker} | {ex.status} | {ex.reason} | {ex.exact_flow_note} |"
            )
    lines.append("")
    lines.append("## Other exclusions")
    lines.append("")
    lines.append("| Ticker | Reason | Exact Flow |")
    lines.append("|---|---|---|")
    for ex in exclusions[:20]:
        if ex.ticker in focus_map:
            continue
        lines.append(f"| {ex.ticker} | {ex.reason} | {ex.exact_flow_note} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_date = dt.date.fromisoformat(args.date)
    base_dir = Path(args.base_dir).expanduser().resolve()
    day_dir = base_dir / run_date.isoformat()
    chain_csv = Path(args.chain_csv).expanduser().resolve() if args.chain_csv else day_dir / f"chain-oi-changes-{run_date.isoformat()}.csv"
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else day_dir / f"morning-watch-setups-{run_date.isoformat()}.md"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else day_dir / f"morning-watch-setups-{run_date.isoformat()}.csv"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = _load_chain(chain_csv, min_dte=int(args.min_dte), max_dte=int(args.max_dte))
    focus_tickers = [t.strip().upper() for t in str(args.focus_tickers).split(",") if t.strip()]
    if df.empty:
        raise SystemExit(f"No chain rows found in DTE range for {chain_csv}")

    svc = SchwabLiveDataService(SchwabAuthConfig.from_env())

    grouped_rows: Dict[str, pd.DataFrame] = {ticker: rows.copy() for ticker, rows in df.groupby("ticker", sort=False)}
    ranked_tickers = sorted(
        grouped_rows.keys(),
        key=lambda ticker: _rank_ticker_summary(_ticker_summary(grouped_rows[ticker])),
        reverse=True,
    )
    shortlist_count = max(int(args.limit) * 4, 30)
    selected_tickers: List[str] = []
    for ticker in [*focus_tickers, *ranked_tickers]:
        if ticker not in selected_tickers:
            selected_tickers.append(ticker)
        if len(selected_tickers) >= shortlist_count and all(t in selected_tickers for t in focus_tickers):
            break

    included: List[Candidate] = []
    exclusions: List[Exclusion] = []
    for ticker in selected_tickers:
        rows = grouped_rows.get(ticker)
        if rows is None:
            continue
        candidate, exclusion = _build_candidate_for_ticker(svc, ticker, rows)
        if candidate is not None:
            included.append(candidate)
        elif exclusion is not None:
            exclusions.append(exclusion)

    included.sort(key=_rank_candidate, reverse=True)
    if focus_tickers:
        included_focus = [c for c in included if c.ticker in focus_tickers]
        included_nonfocus = [c for c in included if c.ticker not in focus_tickers]
        included = included_focus + included_nonfocus
    included = included[: max(1, int(args.limit))]

    included_rows = [
        {
            "ticker": c.ticker,
            "direction": c.direction,
            "strategy": c.strategy,
            "expiry": c.expiry.isoformat(),
            "dte": c.dte,
            "spot": c.spot,
            "lead_symbol": c.lead_symbol,
            "pair_symbol": c.pair_symbol,
            "target_value": c.target_value,
            "stretch_value": c.stretch_value,
            "breakeven": c.breakeven,
            "move_needed_pct": c.move_needed_pct,
            "flow_conviction": c.flow_conviction,
            "flow_conviction_label": c.flow_conviction_label,
            "geometry_label": c.geometry_label,
            "include_reason": c.include_reason,
            "exact_flow_note": c.exact_flow_note,
        }
        for c in included
    ]
    pd.DataFrame(included_rows).to_csv(output_csv, index=False)

    focus_audit: List[Exclusion] = []
    included_map = {c.ticker: c for c in included}
    exclusion_map = {x.ticker: x for x in exclusions}
    for ticker in focus_tickers:
        if ticker in included_map:
            continue
        if ticker in exclusion_map:
            focus_audit.append(exclusion_map[ticker])
        elif ticker in set(df["ticker"]):
            focus_audit.append(Exclusion(ticker=ticker, status="EXCLUDED", reason="Ticker was present but did not survive coded candidate construction."))
        else:
            focus_audit.append(Exclusion(ticker=ticker, status="NOT_FOUND", reason="Ticker not present in the chain-oi file."))

    markdown = _render_markdown(run_date, included, exclusions, focus_audit)
    output_md.write_text(markdown, encoding="utf-8")

    print(f"[ok] wrote {output_md}")
    print(f"[ok] wrote {output_csv}")
    for c in included:
        if "Credit" in c.strategy:
            target_label = f"{c.target_value:.2f} cr"
        else:
            target_label = f"{c.target_value:.2f} db"
        print(
            f"[include] {c.ticker} {c.strategy} {c.expiry.isoformat()} "
            f"{c.lead_strike:g}/{c.pair_strike:g} target={target_label} "
            f"conviction={c.flow_conviction} move={c.move_needed_pct:.2f}%"
        )
    if focus_audit:
        for ex in focus_audit:
            print(f"[focus] {ex.ticker}: {ex.status} - {ex.reason}")
    for c in included:
        if c.ticker in focus_tickers:
            print(f"[focus] {c.ticker}: INCLUDED - {c.include_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
