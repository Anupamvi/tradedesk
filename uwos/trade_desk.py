#!/usr/bin/env python3
"""Trade Desk report: Schwab open-position review with HOLD/CLOSE/ROLL verdicts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uwos.paths import project_root
from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
from uwos.schwab_position_analyzer import analyze_positions
from uwos.trade_monitor import classify_position, compute_verdict, position_key, safe
from uwos.spread_positions import build_position_review_items, compute_spread_verdict


ACTION_PRIORITY = {"CLOSE": 0, "ROLL": 1, "ASSESS": 2, "HOLD": 3}
REPORT_ACTION_PRIORITY = {"CLOSE": 0, "ROLL": 1, "SET STOP": 2, "HOLD": 3}
ACTION_MARKERS = {"CLOSE": "🔴", "ROLL": "🔵", "SET STOP": "🟡", "HOLD": "🟢"}


def _fmt_money(value: Any) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    if abs(v) < 0.5:
        v = 0.0
    return f"${v:,.0f}"


def _fmt_price(value: Any) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    return f"${v:,.2f}"


def _fmt_pct(value: Any) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    return f"{v:+.1f}%"


def _fmt_num(value: Any, places: int = 0) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    return f"{v:.{places}f}"


def _fmt_qty(value: Any) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _fmt_strike(value: Any) -> str:
    v = safe(value, None)
    if v is None:
        return ""
    return f"${v:g}"


def _text(value: Any, fallback: str = "-") -> str:
    text = "" if value is None else str(value).strip()
    return text.replace("\n", " ") if text else fallback


def _wrap_field(label: str, value: Any, *, width: int = 104) -> List[str]:
    prefix = f"{label}: "
    text = _text(value)
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix.rstrip()]


def _action_marker(action: Any) -> str:
    return ACTION_MARKERS.get(str(action or "").upper(), "⚪")


def _action_label(action: Any) -> str:
    text = str(action or "UNKNOWN").upper()
    return f"{_action_marker(text)} {text}"


def _compact_legs(position: Any) -> str:
    text = _text(position)
    return text if len(text) <= 128 else text[:125] + "..."


def _option_leg_description(pos: Dict[str, Any], *, abs_qty: Optional[Any] = None) -> str:
    qty = safe(pos.get("qty"), 0.0)
    qty_abs = safe(abs_qty, None)
    if qty_abs is None:
        qty_abs = abs(qty)
    side = "Short" if qty < 0 else "Long"
    underlying = _text(pos.get("underlying") or str(pos.get("symbol", "")).split()[0])
    expiry = _text(pos.get("expiry"))
    strike = _fmt_strike(pos.get("strike"))
    right = _text(pos.get("put_call")).upper()
    return f"{side} {_fmt_qty(qty_abs)} {underlying} {expiry} {strike} {right}".strip()


def _spread_legs_description(group: Dict[str, Any], qty: Any) -> str:
    short_desc = _option_leg_description(group["short_leg"], abs_qty=qty)
    long_desc = _option_leg_description(group["long_leg"], abs_qty=qty)
    if group.get("net_type") == "debit":
        return f"{long_desc} / {short_desc}"
    return f"{short_desc} / {long_desc}"


def _format_deadline(row: Dict[str, Any]) -> str:
    expiry_text = _text(row.get("expiry"), "")
    dte = safe(row.get("_raw_dte"), None)
    if not expiry_text or dte is None:
        return "at the next market close"
    try:
        expiry = dt.date.fromisoformat(expiry_text)
    except ValueError:
        return "at the next market close"

    as_of = expiry - dt.timedelta(days=int(round(dte)))
    deadline = as_of
    while deadline.weekday() >= 5:
        deadline += dt.timedelta(days=1)
    return f"by market close (4:00 PM ET) on {deadline.strftime('%A, %B')} {deadline.day}, {deadline.year}"


def _action_instruction(row: Dict[str, Any]) -> str:
    action = row.get("action") or row.get("verdict")
    kind = row.get("_kind")
    category = str(row.get("category", "")).lower()
    is_spread = kind == "SPREAD"
    pct_max = safe(row.get("_raw_pct_max"), None)
    pnl_pct = safe(row.get("_raw_pnl_pct"), None)
    spot = safe(row.get("_raw_spot"), None)

    if action == "CLOSE":
        target = "the whole spread as one order; do not leg out" if is_spread else "this option position"
        if "credit" in category and pct_max is not None and pct_max >= 75:
            return (
                f"Close {target}; most max profit is already captured and the remaining premium is mostly "
                "gamma/assignment risk."
            )
        if "debit" in category and pnl_pct is not None and pnl_pct <= -60:
            return f"Close {target}; theta decay makes recovery unlikely from this loss."
        return f"Close {target}."
    if action == "ROLL":
        return (
            "Roll the whole spread as one order; if the roll does not improve risk/reward, close it."
            if is_spread
            else "Roll this option; if you cannot roll for acceptable risk/reward, close it."
        )
    if action == "SET STOP":
        long_strike = safe(row.get("_long_strike"), None)
        short_strike = safe(row.get("_short_strike"), None)
        deadline = _format_deadline(row)
        if "bull call debit" in category and long_strike is not None:
            if spot is not None and short_strike is not None:
                midpoint = (long_strike + short_strike) / 2.0
                if spot >= short_strike:
                    return (
                        f"Set a stop: take profit now, or close the whole spread if the stock cannot hold "
                        f"{_fmt_strike(short_strike)} {deadline}; max profit is capped, so do not leave gains "
                        "exposed to theta/gamma decay."
                    )
                if spot > long_strike:
                    if spot >= midpoint:
                        return (
                            f"Set a stop: protect gains at the spread midpoint; close the whole spread if the stock "
                            f"falls back below {_fmt_strike(midpoint)} {deadline}. If it reaches "
                            f"{_fmt_strike(short_strike)} before then, take profit rather than waiting for expiration."
                        )
                    return (
                        f"Set a stop: close the whole spread if the stock cannot stay above "
                        f"{_fmt_strike(long_strike)} {deadline}; theta/gamma decay is working against the position, "
                        "so do not roll unless the bullish thesis is still strong."
                    )
            return (
                f"Set a stop: close the whole spread if the stock cannot reclaim {_fmt_strike(long_strike)} {deadline}; "
                "theta/gamma decay is working against the position, so do not roll unless the bullish thesis is still strong."
            )
        if "bear put debit" in category and long_strike is not None:
            if spot is not None and short_strike is not None:
                midpoint = (long_strike + short_strike) / 2.0
                if spot <= short_strike:
                    return (
                        f"Set a stop: take profit now, or close the whole spread if the stock cannot stay below "
                        f"{_fmt_strike(short_strike)} {deadline}; max profit is capped, so do not leave gains exposed "
                        "to theta/gamma decay."
                    )
                if spot < long_strike:
                    if spot <= midpoint:
                        return (
                            f"Set a stop: protect gains at the spread midpoint; close the whole spread if the stock "
                            f"rebounds above {_fmt_strike(midpoint)} {deadline}. If it reaches "
                            f"{_fmt_strike(short_strike)} before then, take profit rather than waiting for expiration."
                        )
                    return (
                        f"Set a stop: close the whole spread if the stock cannot stay below "
                        f"{_fmt_strike(long_strike)} {deadline}; theta/gamma decay is working against the position, "
                        "so do not roll unless the bearish thesis is still strong."
                    )
            return (
                f"Set a stop: close the whole spread if the stock cannot break below {_fmt_strike(long_strike)} {deadline}; "
                "theta/gamma decay is working against the position, so do not roll unless the bearish thesis is still strong."
            )
        if "credit" in category and short_strike is not None:
            return (
                f"Set a stop at the short strike {_fmt_strike(short_strike)}; if breached, roll the whole position or close it."
            )
        if "debit" in category:
            return "Set a stop; close if the option remains OTM inside 14 DTE or the loss reaches about 60%."
        return "Set a stop/exit trigger; close or roll only when that trigger hits."
    return "Hold; no action right now."


def _report_action(row: Dict[str, Any]) -> str:
    verdict = row.get("verdict")
    if verdict in {"CLOSE", "ROLL", "HOLD"}:
        return str(verdict)

    reason = str(row.get("reason", "")).lower()
    category = str(row.get("category", "")).lower()
    if "expiration" in reason or "assignment" in reason or "pin risk" in reason:
        return "CLOSE"
    if "itm" in reason or "roll" in reason:
        return "ROLL"
    if "debit" in category or "credit" in category:
        return "SET STOP"
    return "SET STOP"


def _finalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    action = _report_action(row)
    row["action"] = action
    row["instruction"] = _action_instruction(row)
    row["_priority"] = REPORT_ACTION_PRIORITY.get(action, 9)
    return row


def _row_card(row: Dict[str, Any]) -> List[str]:
    details = [
        f"type {row['category']}",
        f"qty {_text(row.get('qty'))}",
    ]
    expiry = _text(row.get("expiry"), "")
    dte = _text(row.get("dte"), "")
    if expiry or dte:
        details.append(f"expiry {expiry or '-'} / DTE {dte or '-'}")
    strike = _text(row.get("strike"), "")
    spot = _text(row.get("underlying_price"), "")
    if strike or spot:
        details.append(f"strike {strike or '-'} / spot {spot or '-'}")

    risk = [
        f"P&L {_text(row.get('pnl'))}",
        f"P&L% {_text(row.get('pnl_pct'))}",
        f"max {_text(row.get('pct_max'))}",
    ]
    theta = _text(row.get("theta_day"), "")
    if theta:
        risk.append(f"theta/day {theta}")
    gamma = _text(row.get("gamma_risk"), "")
    if gamma:
        risk.append(f"gamma {gamma}")
    delta = _text(row.get("delta"), "")
    if delta:
        risk.append(f"delta {delta}")
    entry = _text(row.get("entry_date"), "")
    if entry:
        risk.append(f"entry {entry}")

    lines = [
        f"{_action_marker(row.get('action'))} {row['underlying']} | {row['action']}",
        "",
    ]
    lines.extend(_wrap_field("Do this", row.get("instruction")))
    lines.extend(_wrap_field("Why", row.get("reason")))
    lines.extend(_wrap_field("Legs", _compact_legs(row.get("position"))))
    lines.extend(_wrap_field("Details", "; ".join(details)))
    lines.extend(_wrap_field("Risk", "; ".join(risk)))
    return lines


def _recommendation_cards(rows: List[Dict[str, Any]], *, group_by_verdict: bool) -> str:
    if not rows:
        return "_none_"

    lines: List[str] = []
    if group_by_verdict:
        for verdict in ("CLOSE", "ROLL", "SET STOP"):
            subset = [r for r in rows if r.get("action") == verdict]
            if not subset:
                continue
            lines.extend([_action_label(verdict), ""])
            for row in subset:
                lines.extend(_row_card(row))
                lines.append("")
    else:
        for row in rows:
            lines.extend(_row_card(row))
            lines.append("")
    return "\n".join(lines).rstrip()


def build_recommendations(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    positions = [p for p in (result.get("positions", []) or []) if p.get("asset_type") == "OPTION"]
    rows: List[Dict[str, Any]] = []
    for item in build_position_review_items(positions):
        if item["kind"] == "SPREAD":
            verdict, reason, metrics = compute_spread_verdict(item["group"])
            group = item["group"]
            legs = _spread_legs_description(group, metrics["qty"])
            rows.append(
                _finalize_row({
                    "verdict": verdict,
                    "category": metrics["strategy"],
                    "position": legs,
                    "symbol": item["key"],
                    "underlying": metrics["underlying"],
                    "qty": _fmt_num(metrics["qty"], 0),
                    "expiry": metrics["expiry"],
                    "dte": _fmt_num(metrics["dte"], 0),
                    "strike": metrics["strike_pair"],
                    "underlying_price": _fmt_price(metrics["underlying_price"]),
                    "pnl": _fmt_money(metrics["unrealized_pnl"]),
                    "pnl_pct": _fmt_pct(metrics["pnl_on_risk_pct"]),
                    "pct_max": _fmt_pct(metrics["pct_of_max_profit"]),
                    "delta": _fmt_num(metrics["short_delta"], 2),
                    "theta_day": _fmt_money(metrics["theta_pnl_per_day"]),
                    "gamma_risk": _fmt_num(metrics["gamma_risk"], 1),
                    "entry_date": metrics["entry_date"],
                    "reason": reason,
                    "_priority": ACTION_PRIORITY.get(verdict, 9),
                    "_kind": "SPREAD",
                    "_net_type": metrics["net_type"],
                    "_put_call": metrics["put_call"],
                    "_short_strike": metrics["short_strike"],
                    "_long_strike": metrics["long_strike"],
                    "_raw_dte": metrics["dte"],
                    "_raw_spot": metrics["underlying_price"],
                    "_raw_pct_max": metrics["pct_of_max_profit"],
                    "_raw_pnl_pct": metrics["pnl_on_risk_pct"],
                    "_raw_theta_day": metrics["theta_pnl_per_day"],
                    "_raw_gamma_risk": metrics["gamma_risk"],
                })
            )
            continue

        pos = item["position"]
        key = position_key(pos) or str(pos.get("symbol") or "UNKNOWN").strip() or "UNKNOWN"
        category = classify_position(pos)
        verdict, reason = compute_verdict(pos)
        computed = pos.get("computed", {}) or {}
        quote = pos.get("underlying_quote", {}) or {}
        greeks = pos.get("greeks", {}) or {}
        rows.append(
            _finalize_row({
                "verdict": verdict,
                "category": category,
                "position": _option_leg_description(pos),
                "symbol": key,
                "underlying": pos.get("underlying") or key,
                "qty": pos.get("qty", ""),
                "expiry": pos.get("expiry") or "",
                "dte": _fmt_num(computed.get("dte"), 0),
                "strike": _fmt_price(pos.get("strike")),
                "underlying_price": _fmt_price(quote.get("last")),
                "pnl": _fmt_money(computed.get("unrealized_pnl")),
                "pnl_pct": _fmt_pct(computed.get("unrealized_pnl_pct")),
                "pct_max": _fmt_pct(computed.get("pct_of_max_profit")),
                "delta": _fmt_num(greeks.get("delta"), 2),
                "theta_day": _fmt_money(computed.get("theta_pnl_per_day")),
                "gamma_risk": _fmt_num(computed.get("gamma_risk"), 1),
                "entry_date": pos.get("entry_date") or "",
                "reason": reason,
                "_priority": ACTION_PRIORITY.get(verdict, 9),
                "_kind": "POSITION",
                "_put_call": pos.get("put_call", ""),
                "_short_strike": pos.get("strike") if pos.get("qty", 0) < 0 else None,
                "_long_strike": pos.get("strike") if pos.get("qty", 0) > 0 else None,
                "_raw_dte": computed.get("dte"),
                "_raw_spot": quote.get("last"),
                "_raw_pct_max": computed.get("pct_of_max_profit"),
                "_raw_pnl_pct": computed.get("unrealized_pnl_pct"),
                "_raw_theta_day": computed.get("theta_pnl_per_day"),
                "_raw_gamma_risk": computed.get("gamma_risk"),
            })
        )
    rows.sort(key=lambda r: (r["_priority"], r["underlying"], r["symbol"]))
    return rows


def build_report(
    result: Dict[str, Any],
    recommendation_rows: List[Dict[str, Any]],
    *,
    days: int,
    symbol: Optional[str],
    json_path: Path,
) -> str:
    as_of = result.get("as_of", "")
    positions = result.get("positions", []) or []
    option_positions = [p for p in positions if p.get("asset_type") == "OPTION"]
    omitted_count = result.get("omitted_non_option_positions")
    if omitted_count is None:
        omitted_count = len(positions) - len(option_positions)
    sources = result.get("context_sources", {}) or {}
    spread_rows = [r for r in recommendation_rows if r.get("_kind") == "SPREAD"]
    actionable = [r for r in recommendation_rows if r.get("action") in {"CLOSE", "ROLL", "SET STOP"}]
    hold = [r for r in recommendation_rows if r.get("action") == "HOLD"]
    counts: Dict[str, int] = {}
    for r in recommendation_rows:
        action = r.get("action", r.get("verdict", "UNKNOWN"))
        counts[action] = counts.get(action, 0) + 1

    title = f"Trade Desk Recommendations - {dt.date.today().isoformat()}"
    source_text = ", ".join(
        [
            f"positions={sources.get('positions', 'schwab')}",
            f"transactions={sources.get('transactions', 'schwab')}",
            f"quotes={sources.get('quotes', 'schwab')}",
            f"option chains={sources.get('option_chains', 'schwab')}",
            f"yfinance={'on' if sources.get('external_yfinance') else 'off'}",
        ]
    )
    verdict_mix = ", ".join(
        f"{_action_label(k)}={counts.get(k, 0)}" for k in ("CLOSE", "ROLL", "SET STOP", "HOLD")
    )
    lines = [
        title.upper(),
        "",
        f"As of: {as_of}",
        f"Schwab history lookback: {days} days",
        f"Symbol filter: {symbol.upper() if symbol else 'all'}",
        f"Open option legs reviewed: {len(option_positions)}",
        f"Equities/funds omitted: {omitted_count}",
        f"Spread groups reviewed: {len(spread_rows)}",
    ]
    lines.extend(_wrap_field("Data sources", source_text))
    lines.extend(_wrap_field("Verdict mix", verdict_mix))
    lines.extend(_wrap_field("Position JSON", json_path))
    lines.extend([
        "",
        "ACTION REQUIRED",
        "",
        _recommendation_cards(actionable, group_by_verdict=True),
        "",
        "🟢 KEEP / HOLD",
        "",
        _recommendation_cards(hold, group_by_verdict=False),
        "",
        "NOTES",
        "HOLD means keep for now under the current rule engine.",
        "SET STOP means keep only with a clear exit trigger; it is not a blank-check hold.",
        "Spread rows are evaluated as one defined-risk position; CLOSE or ROLL means act on both legs together, not one leg.",
        "This report defaults to Schwab API data only. Use --with-yfinance-context only if you explicitly want Yahoo/yfinance enrichment.",
        "Unusual Whales/browser research should be a separate follow-up layer after this position review, not a hidden data source.",
        "This command reads Schwab data and writes reports only. It does not place trades.",
        "",
    ])
    return "\n".join(lines)


def run_trade_desk(
    *,
    days: int,
    symbol: Optional[str],
    out_dir: Path,
    account_index: int,
    manual_auth: bool,
    with_yfinance_context: bool = False,
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, manual_auth=manual_auth, interactive_login=False)
    result = analyze_positions(
        svc=svc,
        days=days,
        account_index=account_index,
        symbol_filter=symbol,
        include_yfinance=with_yfinance_context,
    )
    all_positions = result.get("positions", []) or []
    option_positions = [p for p in all_positions if p.get("asset_type") == "OPTION"]
    result = {
        **result,
        "positions": option_positions,
        "omitted_non_option_positions": len(all_positions) - len(option_positions),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    suffix = f"_{symbol.upper()}" if symbol else ""
    json_path = out_dir / f"position_data_{today}{suffix}.json"
    report_path = out_dir / f"trade-desk-{today}{suffix}.md"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    recommendations = build_recommendations(result)
    report = build_report(
        result,
        recommendations,
        days=days,
        symbol=symbol,
        json_path=json_path,
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path, json_path, recommendations


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review open Schwab positions and write a trade-desk recommendation report."
    )
    parser.add_argument("days_positional", nargs="?", type=int, help="History lookback days, e.g. 90.")
    parser.add_argument("symbol_positional", nargs="?", help="Optional underlying filter, e.g. AAPL.")
    parser.add_argument("--days", type=int, default=None, help="History lookback days.")
    parser.add_argument("--symbol", default=None, help="Optional underlying filter.")
    parser.add_argument("--account-index", type=int, default=0, help="Schwab account index.")
    parser.add_argument("--out-dir", default="", help="Output directory.")
    parser.add_argument("--manual-auth", action="store_true", help="Use manual Schwab OAuth flow.")
    parser.add_argument("--with-yfinance-context", action="store_true", help="Opt in to Yahoo/yfinance enrichment.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    days = int(args.days if args.days is not None else (args.days_positional or 90))
    symbol = args.symbol or args.symbol_positional
    root = project_root()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else root / "out" / "trade_analysis"
    report_path, json_path, recommendations = run_trade_desk(
        days=days,
        symbol=symbol,
        out_dir=out_dir.resolve(),
        account_index=int(args.account_index),
        manual_auth=bool(args.manual_auth),
        with_yfinance_context=bool(args.with_yfinance_context),
    )

    counts: Dict[str, int] = {}
    for r in recommendations:
        action = r.get("action", r.get("verdict", "UNKNOWN"))
        counts[action] = counts.get(action, 0) + 1
    print("Trade desk completed.")
    print(f"- Report: {report_path}")
    print(f"- Position JSON: {json_path}")
    print(
        "- Verdict mix: "
        + ", ".join(f"{k}={counts.get(k, 0)}" for k in ("CLOSE", "ROLL", "SET STOP", "HOLD"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
