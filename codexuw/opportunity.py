from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .lifecycle import apply_lifecycle_triggers
from .occ import parse_occ_symbol
from .performance import setup_family
from .pipeline_versions import PIPELINE_NAME_V3, PIPELINE_VERSION_V3


INDEX_ETFS = {
    "DIA",
    "IWM",
    "QQQ",
    "SMH",
    "SOXX",
    "SPY",
    "XBI",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XOP",
}

OPPORTUNITY_COLUMNS = [
    "Lane",
    "Status",
    "Ticker",
    "Trade",
    "Expiry",
    "Entry limit",
    "Live mid/natural",
    "Max profit",
    "Max loss",
    "Target profit",
    "Expected value source",
    "Edge sample size / win rate / avg P/L",
    "Required confirmation",
    "Monitor trigger",
    "Why Execute, Scout, Research, or Avoid",
]

TARGET_TICKET_COLUMNS = [
    "Rank",
    "Lane",
    "Status",
    "Ticker",
    "Trade",
    "Next-session swing entry",
    "Current mid/natural",
    "Profit target",
    "Max loss",
    "Swing trend evidence",
    "Swing work instruction",
    "What blocks entry",
]


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _money(value: object, *, blank: str = "") -> str:
    if isinstance(value, str):
        text = value.replace("$", "").replace(",", "").strip()
        number = safe_float(text)
    else:
        number = safe_float(value)
    return f"${number:,.2f}" if math.isfinite(number) else blank


def _pct(value: object, *, blank: str = "") -> str:
    number = safe_float(value)
    return f"{number:.1%}" if math.isfinite(number) else blank


def _is_debit(row: pd.Series | dict[str, Any]) -> bool:
    text = f"{row.get('strategy', '')} {row.get('direction', '')}".lower()
    return "debit" in text or "bull call" in text or "bear put" in text


def _is_credit(row: pd.Series | dict[str, Any]) -> bool:
    return not _is_debit(row)


def _ticker(row: pd.Series | dict[str, Any]) -> str:
    return _clean(row.get("ticker")).upper()


def _leg_label(symbol: object) -> str:
    parsed = parse_occ_symbol(symbol)
    if parsed is None:
        return _clean(symbol)
    return f"{parsed.root} {parsed.expiry} {parsed.strike:g}{parsed.right}"


def _position_text(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    words = text.split()
    replaced: list[str] = []
    idx = 0
    while idx < len(words):
        if idx + 1 < len(words):
            candidate = words[idx] + words[idx + 1]
            parsed = parse_occ_symbol(candidate)
            if parsed is not None:
                replaced.append(_leg_label(candidate))
                idx += 2
                continue
        parsed = parse_occ_symbol(words[idx])
        replaced.append(_leg_label(words[idx]) if parsed is not None else words[idx])
        idx += 1
    return " ".join(replaced)


def _lane_for_candidate(row: pd.Series | dict[str, Any]) -> str:
    ticker = _ticker(row)
    if bool(row.get("index_fallback", False)) or ticker in INDEX_ETFS:
        return "Index/ETF"
    if _is_debit(row):
        return "Momentum Debit"
    if _clean(row.get("trade_status")) == "Watch" or "scout" in _clean(row.get("trade_tier")).lower():
        return "Scout"
    strategy = f"{row.get('strategy', '')} {row.get('direction', '')}".lower()
    if "cash-secured" in strategy or "csp" in strategy or "wheel" in strategy:
        return "Wheel/Cash"
    return "Execute" if _clean(row.get("trade_status")) == "Execute" else "Research/Avoid"


def _status_for_candidate(row: pd.Series | dict[str, Any]) -> str:
    status = _clean(row.get("trade_status"))
    tier = _clean(row.get("trade_tier")).lower()
    if status == "Execute":
        return "🟢 Execute"
    if status == "Watch" and ("work-limit" in tier or "near-trigger" in tier or "price" in tier):
        return "🔵 Scout/Work Limit"
    if status == "Watch" or "scout" in tier:
        return "🔵 Scout"
    if status == "Avoid":
        return "🔴 Avoid"
    return "🟡 Research"


def _trade_text(row: pd.Series | dict[str, Any]) -> str:
    strategy = _clean(row.get("strategy")) or _clean(row.get("direction")) or "Option structure"
    short = _clean(row.get("short_leg")) or _clean(row.get("sell_leg"))
    long = _clean(row.get("long_leg")) or _clean(row.get("buy_leg"))
    if short or long:
        short_label = _leg_label(short) if short else "n/a"
        long_label = _leg_label(long) if long else "n/a"
        if _is_debit(row):
            return f"{strategy}: buy {long_label} / sell {short_label}"
        return f"{strategy}: sell {short_label} / buy {long_label}"
    return strategy


def _entry_limit(row: pd.Series | dict[str, Any]) -> str:
    required = safe_float(row.get("required_entry"))
    if not math.isfinite(required):
        required = safe_float(row.get("entry_limit_credit"))
    if not math.isfinite(required):
        required = safe_float(row.get("entry_limit_debit"))
    if not math.isfinite(required):
        required = safe_float(row.get("credit") if _is_credit(row) else row.get("debit"))
    if not math.isfinite(required):
        return "fresh Schwab recheck"
    return f">= ${required:.2f} credit" if _is_credit(row) else f"<= ${required:.2f} debit"


def _live_mid_natural(row: pd.Series | dict[str, Any]) -> str:
    if _is_debit(row):
        mid = safe_float(row.get("mid_debit"), safe_float(row.get("debit")))
        natural = safe_float(row.get("natural_debit"))
        if not math.isfinite(natural):
            natural = safe_float(row.get("debit"))
    else:
        mid = safe_float(row.get("mid_credit"), safe_float(row.get("credit")))
        natural = safe_float(row.get("natural_credit"))
        if not math.isfinite(natural):
            natural = safe_float(row.get("credit"))
    if math.isfinite(mid) and math.isfinite(natural):
        return f"{mid:.2f} / {natural:.2f}"
    if math.isfinite(mid):
        return f"{mid:.2f} / recheck"
    return "fresh Schwab recheck"


def _fill_ladder(row: pd.Series | dict[str, Any]) -> str:
    entry = _entry_limit(row)
    width = _pct(row.get("quote_width_pct"), blank="unknown width")
    if "fresh Schwab recheck" in entry:
        return "Reprice in Schwab first; cancel if bid/ask is too wide or only natural fill is realistic."
    if _is_debit(row):
        return (
            f"Start at {entry}; do not chase above target debit; cancel if quote width is {width} "
            "or fresh natural is the only realistic fill."
        )
    return (
        f"Start at {entry}; do not accept less than target credit; cancel if quote width is {width} "
        "or fresh natural is the only realistic fill."
    )


def _target_profit(row: pd.Series | dict[str, Any]) -> float:
    explicit = safe_float(row.get("target_profit_total"))
    if math.isfinite(explicit):
        return explicit
    max_profit = safe_float(row.get("max_profit"))
    if not math.isfinite(max_profit):
        return math.nan
    return max_profit * (0.45 if _is_debit(row) else 0.60)


def _edge_text(row: pd.Series | dict[str, Any]) -> str:
    size = safe_float(row.get("edge_sample_size"))
    win = safe_float(row.get("edge_win_rate"))
    avg = safe_float(row.get("edge_avg_pnl"))
    parts = []
    parts.append(f"n={int(size)}" if math.isfinite(size) else "n=unavailable")
    parts.append(f"win={win:.0%}" if math.isfinite(win) else "win=unavailable")
    parts.append(f"avg={_money(avg, blank='unavailable')}")
    return " / ".join(parts)


def _ev_source(row: pd.Series | dict[str, Any]) -> str:
    live = _clean(row.get("live_status"))
    edge = _clean(row.get("edge_verdict")) or _clean(row.get("replay_ev_verdict"))
    source = _clean(row.get("edge_match_level")) or _clean(row.get("construction_source"))
    bits = []
    if live:
        bits.append(f"Schwab {live}")
    if edge:
        bits.append(f"edge {edge}")
    if source:
        bits.append(source)
    return "; ".join(bits) if bits else "not established"


def _confirmation(row: pd.Series | dict[str, Any]) -> str:
    improve = _clean(row.get("what_must_improve"))
    failed = _clean(row.get("confirmations_failed"))
    primary = _clean(row.get("primary_blocker"))
    if improve:
        return improve
    if failed:
        return failed
    if primary:
        return primary
    if _clean(row.get("trade_status")) == "Execute":
        return "fresh Schwab quote, portfolio risk cap, and catalyst check still pass before manual order entry"
    return "manual confirmation required before entry"


def _why(row: pd.Series | dict[str, Any]) -> str:
    reason = _clean(row.get("trade_status_reason"))
    hard = _clean(row.get("hard_rejects"))
    penalties = _clean(row.get("penalties"))
    if reason:
        return reason
    if hard:
        return hard
    if penalties:
        return penalties
    status = _clean(row.get("trade_status"))
    if status == "Execute":
        return "live pricing, liquidity, edge, and risk gates passed"
    return "no higher-quality validated setup surfaced"


def _trend_evidence(row: pd.Series | dict[str, Any]) -> str:
    bits: list[str] = []
    direction = _clean(row.get("direction"))
    if direction:
        bits.append(direction)
    flow = _clean(row.get("flow_quality"))
    if flow:
        bits.append(f"flow={flow}")
    oi = _clean(row.get("oi_carryover_status"))
    if oi:
        bits.append(f"OI={oi}")
    edge = _clean(row.get("edge_verdict")) or _clean(row.get("replay_ev_verdict"))
    if edge:
        bits.append(f"edge={edge}")
    confirmation = safe_float(row.get("confirmation_score"))
    if math.isfinite(confirmation):
        bits.append(f"confirm={confirmation:.1f}/10")
    score = safe_float(row.get("score"))
    if math.isfinite(score):
        bits.append(f"score={score:.1f}")
    reason = _clean(row.get("flow_quality_reason"))
    if reason and len(bits) < 5:
        bits.append(reason)
    return "; ".join(bits)


def _candidate_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    status = _status_for_candidate(row)
    return {
        "Lane": _lane_for_candidate(row),
        "Status": status,
        "Ticker": _ticker(row),
        "Trade": _trade_text(row),
        "Expiry": _clean(row.get("expiry")) or _clean(row.get("expiration_date")),
        "Entry limit": _entry_limit(row),
        "Live mid/natural": _live_mid_natural(row),
        "Max profit": _money(row.get("max_profit")),
        "Max loss": _money(row.get("max_loss")),
        "Target profit": _money(_target_profit(row)),
        "Expected value source": _ev_source(row),
        "Edge sample size / win rate / avg P/L": _edge_text(row),
        "Required confirmation": _confirmation(row),
        "Monitor trigger": "",
        "Why Execute, Scout, Research, or Avoid": _why(row),
        "EOD trend evidence": _trend_evidence(row),
        "fill_ladder": _fill_ladder(row),
        "quote_width_pct": row.get("quote_width_pct"),
        "ticker": _ticker(row),
        "strategy": _clean(row.get("strategy")),
        "direction": _clean(row.get("direction")),
        "trade_status": _clean(row.get("trade_status")),
        "trade_tier": _clean(row.get("trade_tier")),
        "expiry": row.get("expiry"),
        "dte": row.get("dte"),
        "credit": row.get("credit"),
        "debit": row.get("debit"),
        "short_strike": row.get("short_strike"),
        "short_delta": row.get("short_delta"),
        "max_profit": row.get("max_profit"),
        "max_loss": row.get("max_loss"),
        "recommended_limit": safe_float(row.get("required_entry")),
        "edge_sample_size": row.get("edge_sample_size"),
        "edge_win_rate": row.get("edge_win_rate"),
        "edge_avg_pnl": row.get("edge_avg_pnl"),
        "actual_fill": "",
        "close_fill": "",
        "realized_pnl": "",
        "mfe": "",
        "mae": "",
        "thesis_worked": "",
        "reason_for_win_loss": "",
        "slippage_vs_recommendation": "",
        "monitor_triggered": "",
    }


def _target_work_instruction(row: pd.Series | dict[str, Any]) -> str:
    status = _clean(row.get("Status"))
    entry = _clean(row.get("Entry limit"))
    if "Execute" in status:
        return f"Manual ticket can be worked at {entry} after fresh Schwab risk/news check."
    if "Work Limit" in status:
        return f"NOT AN ORDER - target only at {entry}; fresh re-score required before entry."
    if "Scout" in status:
        return f"1-lot only at {entry} after listed confirmation clears."
    if "Repair" in status:
        return "Manage or reduce existing risk before adding new unrelated risk."
    return f"Target only at {entry}; blocker must clear before entry."


def build_target_ticket_board(board: pd.DataFrame, *, max_rows: int | None = 0) -> pd.DataFrame:
    """Compact EOD swing target sheet for the next session.

    This is deliberately broader than Execute. It keeps targetable Work Limit,
    Scout, Momentum Debit, Index/ETF, Wheel/Cash, and selected Research rows visible
    as manual swing order-ticket candidates with exact prices.
    """
    if board.empty:
        return pd.DataFrame(columns=TARGET_TICKET_COLUMNS)
    source = board.copy()
    status_text = source["Status"].astype(str)
    trade_text = source["Trade"].astype(str)
    targetish = (
        source["Entry limit"].astype(str).str.contains(r"\$", regex=True, na=False)
        & ~status_text.str.contains("Blocked|Avoid", regex=True, na=False)
        & ~trade_text.str.startswith("No ", na=False)
    )
    if "edge_avg_pnl" in source.columns:
        edge_avg = pd.to_numeric(source["edge_avg_pnl"], errors="coerce")
        targetish &= ~(edge_avg <= 0)
    source = source[targetish].copy()
    if source.empty:
        return pd.DataFrame(columns=TARGET_TICKET_COLUMNS)
    status_rank = status_text.loc[source.index].map(
        lambda text: 5 if "Execute" in text else 4 if "Work Limit" in text else 3 if "Scout" in text else 2 if "Research" in text else 1
    )
    lane_rank = source["Lane"].astype(str).map(
        {
            "Scout": 6,
            "Momentum Debit": 5,
            "Index/ETF": 4,
            "Wheel/Cash": 3,
            "Portfolio Repair": 2,
            "Research/Avoid": 1,
        }
    ).fillna(0)
    source["_target_rank"] = status_rank + lane_rank / 10.0
    source = source.sort_values(["_target_rank", "Target profit"], ascending=[False, False])
    if max_rows and max_rows > 0:
        source = source.head(int(max_rows))
    rows = []
    for rank, (_, row) in enumerate(source.iterrows(), start=1):
        rows.append(
            {
                "Rank": rank,
                "Lane": row.get("Lane", ""),
                "Status": row.get("Status", ""),
                "Ticker": row.get("Ticker", ""),
                "Trade": row.get("Trade", ""),
                "Next-session swing entry": row.get("Entry limit", ""),
                "Current mid/natural": row.get("Live mid/natural", ""),
                "Profit target": row.get("Target profit", ""),
                "Max loss": row.get("Max loss", ""),
                "Swing trend evidence": row.get("EOD trend evidence", row.get("Expected value source", "")),
                "Swing work instruction": _target_work_instruction(row),
                "What blocks entry": row.get("Required confirmation", "") or row.get("Why Execute, Scout, Research, or Avoid", ""),
            }
        )
    return pd.DataFrame(rows, columns=TARGET_TICKET_COLUMNS)


def _best_rows(scored: pd.DataFrame, mask: pd.Series, *, n: int = 1) -> pd.DataFrame:
    if scored.empty:
        return scored.iloc[0:0].copy()
    part = scored[mask].copy()
    if part.empty:
        return part
    sort_cols = [col for col in ["trade_status", "confirmation_score", "score", "edge_sample_size"] if col in part.columns]
    if "trade_status" in sort_cols:
        status_rank = {"Execute": 4, "Watch": 3, "Research": 2, "Avoid": 1}
        part["_status_rank"] = part["trade_status"].map(status_rank).fillna(0)
        sort_cols = ["_status_rank"] + [c for c in sort_cols if c != "trade_status"]
    return part.sort_values(sort_cols, ascending=False).head(n).drop(columns=["_status_rank"], errors="ignore")


def _portfolio_repair_rows(portfolio: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not portfolio or portfolio.get("status") != "ok":
        return [
            {
                "Lane": "Portfolio Repair",
                "Status": "🔴 Blocked",
                "Ticker": "PORTFOLIO",
                "Trade": "Schwab position review unavailable",
                "Expiry": "",
                "Entry limit": "do not add risk",
                "Live mid/natural": "",
                "Max profit": "",
                "Max loss": "",
                "Target profit": "",
                "Expected value source": "Schwab portfolio pull failed or skipped",
                "Edge sample size / win rate / avg P/L": "",
                "Required confirmation": "restore Schwab positions/orders/fills access",
                "Monitor trigger": "",
                "Why Execute, Scout, Research, or Avoid": "open-risk review must complete before adding unrelated risk",
            }
        ]
    actions = list(portfolio.get("risk_actions") or [])
    if not actions:
        return [
            {
                "Lane": "Portfolio Repair",
                "Status": "🟢 Clear",
                "Ticker": "PORTFOLIO",
                "Trade": "No urgent open-risk repair surfaced",
                "Expiry": "",
                "Entry limit": "n/a",
                "Live mid/natural": "",
                "Max profit": "",
                "Max loss": "",
                "Target profit": "",
                "Expected value source": "Schwab portfolio state",
                "Edge sample size / win rate / avg P/L": "",
                "Required confirmation": "confirm no new fills since snapshot",
                "Monitor trigger": "",
                "Why Execute, Scout, Research, or Avoid": "portfolio review did not outrank new opportunities",
            }
        ]
    rows: list[dict[str, Any]] = []
    priority = {"CLOSE": 5, "REDUCE": 5, "ROLL": 4, "TAKE PROFIT": 4, "SET STOP": 3, "HOLD": 2}
    actions = sorted(actions, key=lambda item: priority.get(str(item.get("action", "")).upper(), 1), reverse=True)
    for action in actions[:3]:
        verb = _clean(action.get("action")) or "REVIEW"
        rows.append(
            {
                "Lane": "Portfolio Repair",
                "Status": "🟡 Repair" if verb not in {"HOLD"} else "🟢 Hold",
                "Ticker": _clean(action.get("ticker")) or "PORTFOLIO",
                "Trade": f"{verb}: {_position_text(action.get('position'))}",
                "Expiry": "",
                "Entry limit": "manual Schwab review",
                "Live mid/natural": "",
                "Max profit": "",
                "Max loss": "",
                "Target profit": "",
                "Expected value source": "Schwab positions/orders/fills",
                "Edge sample size / win rate / avg P/L": "",
                "Required confirmation": _clean(action.get("instruction")),
                "Monitor trigger": _clean(action.get("instruction")),
                "Why Execute, Scout, Research, or Avoid": _clean(action.get("reason")),
            }
        )
    return rows


def _status_rank(status: object) -> int:
    return {"Execute": 4, "Watch": 3, "Research": 2, "Avoid": 1}.get(_clean(status), 0)


def _numeric_series(df: pd.DataFrame, column: str, *, default: float = math.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _wheel_cash_candidate_rows(scored: pd.DataFrame, portfolio: dict[str, Any] | None, *, n: int = 2) -> list[dict[str, Any]]:
    if scored is None or scored.empty or not portfolio or portfolio.get("status") != "ok":
        return []
    cash = safe_float(portfolio.get("cash"))
    if not math.isfinite(cash) or cash <= 0:
        return []
    df = scored.copy()
    if "direction" not in df.columns or "ticker" not in df.columns:
        return []
    df = df[
        df["direction"].astype(str).eq("Bull Put")
        & ~df["ticker"].astype(str).str.upper().isin(INDEX_ETFS)
        & df.get("live_status", pd.Series("", index=df.index)).astype(str).eq("PASS")
    ].copy()
    if df.empty:
        return []
    df["_strike_cash_required"] = _numeric_series(df, "short_strike").fillna(0.0) * 100.0
    df = df[df["_strike_cash_required"].gt(0) & df["_strike_cash_required"].le(cash)].copy()
    if df.empty:
        return []
    df["_quote"] = _numeric_series(df, "quote_width_pct").fillna(9.0)
    df["_liq"] = (
        _numeric_series(df, "short_oi").fillna(0.0)
        + _numeric_series(df, "short_volume").fillna(0.0)
    )
    df["_status_rank"] = df.get("trade_status", pd.Series("", index=df.index)).map(_status_rank).fillna(0)
    df["_wheel_rank"] = (
        df["_status_rank"] * 100.0
        + _numeric_series(df, "score").fillna(0.0)
        + df["_liq"].clip(upper=5000) / 5000.0
        - df["_quote"].clip(upper=1.0)
    )
    rows: list[dict[str, Any]] = []
    for _, row in df.sort_values("_wheel_rank", ascending=False).head(n).iterrows():
        ticker = _ticker(row)
        short_leg = _clean(row.get("short_leg"))
        strike = safe_float(row.get("short_strike"))
        bid = safe_float(row.get("sell_leg_bid"))
        ask = safe_float(row.get("sell_leg_ask"))
        mid = safe_float(row.get("sell_leg_mid"))
        entry = mid if math.isfinite(mid) and mid > 0 else safe_float(row.get("credit"))
        natural = bid if math.isfinite(bid) and bid > 0 else math.nan
        max_profit = entry * 100.0 if math.isfinite(entry) else math.nan
        max_loss = strike * 100.0 - max_profit if math.isfinite(strike) and math.isfinite(max_profit) else math.nan
        expiry = _clean(row.get("expiry"))
        if _clean(row.get("trade_status")) == "Execute" and str(row.get("v3_confirmation_status") or "") == "cleared":
            status = "🟢 Execute"
        elif _clean(row.get("trade_status")) in {"Execute", "Watch", "Research"} and safe_float(row.get("quote_width_pct"), 9.0) <= 0.20:
            status = "🔵 Scout"
        else:
            status = "🟡 Research"
        rows.append(
            {
                "Lane": "Wheel/Cash",
                "Status": status,
                "Ticker": ticker,
                "Trade": f"Cash-secured put (assignment-risk): sell {_leg_label(short_leg) if short_leg else f'{ticker} {expiry} {strike:g}P'}",
                "Expiry": expiry,
                "Entry limit": f">= ${entry:.2f} credit" if math.isfinite(entry) else "fresh Schwab CSP reprice",
                "Live mid/natural": f"{entry:.2f} / {natural:.2f}" if math.isfinite(entry) and math.isfinite(natural) else "fresh Schwab recheck",
                "Max profit": _money(max_profit),
                "Max loss": _money(max_loss),
                "Target profit": _money(max_profit * 0.50 if math.isfinite(max_profit) else math.nan),
                "Expected value source": _ev_source(row) + "; assignment quality/cash budget",
                "Edge sample size / win rate / avg P/L": _edge_text(row),
                "Required confirmation": (
                    "manual assignment-quality check, no near-term earnings, fresh Schwab CSP quote, "
                    "cash still available, and no duplicate ticker exposure"
                ),
                "Monitor trigger": "",
                "Why Execute, Scout, Research, or Avoid": (
                    f"cash budget ${cash:,.0f} can secure strike ${strike:g}; assignment-risk lane, not defined-risk spread"
                    if math.isfinite(strike)
                    else "cash-secured put candidate needs fresh strike/quote validation"
                ),
                "fill_ladder": (
                    f"Start at >= ${entry:.2f} credit; cancel if bid/ask widens or assignment thesis fails."
                    if math.isfinite(entry)
                    else "Reprice in Schwab first."
                ),
                "quote_width_pct": row.get("quote_width_pct"),
                "ticker": ticker,
                "strategy": "Cash-secured put",
                "direction": "Bull Put",
                "trade_status": "Execute" if "Execute" in status else "Watch" if "Scout" in status else "Research",
                "trade_tier": "assignment-risk-wheel",
                "expiry": row.get("expiry"),
                "dte": row.get("dte"),
                "credit": entry,
                "short_strike": strike,
                "short_delta": row.get("short_delta"),
                "max_profit": max_profit,
                "max_loss": max_loss,
                "recommended_limit": entry,
                "edge_sample_size": row.get("edge_sample_size"),
                "edge_win_rate": row.get("edge_win_rate"),
                "edge_avg_pnl": row.get("edge_avg_pnl"),
                "actual_fill": "",
                "close_fill": "",
                "realized_pnl": "",
                "mfe": "",
                "mae": "",
                "thesis_worked": "",
                "reason_for_win_loss": "",
                "slippage_vs_recommendation": "",
                "monitor_triggered": "",
            }
        )
    return rows


def _wheel_cash_row(portfolio: dict[str, Any] | None) -> dict[str, Any]:
    cash = safe_float((portfolio or {}).get("cash"))
    if portfolio and portfolio.get("status") == "ok" and math.isfinite(cash) and cash > 0:
        return {
            "Lane": "Wheel/Cash",
            "Status": "🟡 Research",
            "Ticker": "CASH",
            "Trade": "Cash-secured put / wheel candidate search",
            "Expiry": "",
            "Entry limit": "requires live CSP chain pricing",
            "Live mid/natural": "",
            "Max profit": "",
            "Max loss": f"${cash:,.0f} cash budget ceiling before risk caps",
            "Target profit": "",
            "Expected value source": "Schwab cash balance; assignment quality not yet validated",
            "Edge sample size / win rate / avg P/L": "",
            "Required confirmation": "assignment-quality ticker, valuation/technical support, liquid CSP chain, and cash budget must pass",
            "Monitor trigger": "",
            "Why Execute, Scout, Research, or Avoid": "Wheel/Cash lane is visible, but no validated assignment-risk trade was priced in this run",
        }
    return {
        "Lane": "Wheel/Cash",
        "Status": "🔴 Blocked",
        "Ticker": "CASH",
        "Trade": "Cash-secured put / wheel candidate search",
        "Expiry": "",
        "Entry limit": "blocked",
        "Live mid/natural": "",
        "Max profit": "",
        "Max loss": "",
        "Target profit": "",
        "Expected value source": "Schwab cash unavailable",
        "Edge sample size / win rate / avg P/L": "",
        "Required confirmation": "restore cash/portfolio state before assignment-risk recommendations",
        "Monitor trigger": "",
        "Why Execute, Scout, Research, or Avoid": "cash budget or Schwab portfolio state unavailable",
    }


def build_opportunity_board(
    *,
    scored: pd.DataFrame,
    final: pd.DataFrame,
    watchlist: pd.DataFrame | None,
    portfolio: dict[str, Any] | None,
    max_rows: int | None = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add_frame(frame: pd.DataFrame, limit: int | None = None) -> None:
        selected = frame if limit is None else frame.head(limit)
        for _, row in selected.iterrows():
            board_row = _candidate_row(row)
            key = (
                board_row["Lane"],
                board_row["Status"],
                board_row["Ticker"],
                board_row["Trade"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(board_row)

    active_cap = int(max_rows) if max_rows and max_rows > 0 else None

    if final is not None and not final.empty:
        add_frame(final, active_cap)

    if scored is not None and not scored.empty and "trade_status" in scored.columns:
        add_frame(_best_rows(scored, scored["trade_status"].astype(str).eq("Watch"), n=4))
        single_name_debit = scored.apply(_is_debit, axis=1) & ~scored["ticker"].astype(str).str.upper().isin(INDEX_ETFS)
        add_frame(_best_rows(scored, single_name_debit, n=3))
        add_frame(_best_rows(scored, scored.apply(_is_debit, axis=1), n=3))
        add_frame(_best_rows(scored, scored["ticker"].astype(str).str.upper().isin(INDEX_ETFS) | scored.get("index_fallback", pd.Series(False, index=scored.index)).astype(bool), n=3))
        add_frame(_best_rows(scored, scored["trade_status"].astype(str).isin(["Research", "Avoid"]), n=4))

    if not any(row.get("Lane") == "Portfolio Repair" for row in rows):
        rows.extend(_portfolio_repair_rows(portfolio))
    if not any(row.get("Lane") == "Wheel/Cash" for row in rows):
        wheel_rows = _wheel_cash_candidate_rows(scored, portfolio)
        rows.extend(wheel_rows if wheel_rows else [_wheel_cash_row(portfolio)])

    if not any("Execute" in str(row.get("Status")) for row in rows):
        required_lanes = ["Scout", "Momentum Debit", "Index/ETF", "Portfolio Repair", "Wheel/Cash"]
        present_status = {row.get("Status") for row in rows}
        present_lanes = {row.get("Lane") for row in rows}
        if not any("Scout" in str(status) for status in present_status):
            rows.append(
                {
                    "Lane": "Scout",
                    "Status": "🔴 Blocked",
                    "Ticker": "SCOUT",
                    "Trade": "No 1-lot scout qualified",
                    "Expiry": "",
                    "Entry limit": "blocked",
                    "Live mid/natural": "",
                    "Max profit": "",
                    "Max loss": "",
                    "Target profit": "",
                    "Expected value source": "candidate audit",
                    "Edge sample size / win rate / avg P/L": "",
                    "Required confirmation": "need live pricing/risk edge with only human-confirmation blockers",
                    "Monitor trigger": "",
                    "Why Execute, Scout, Research, or Avoid": "no Watch/manual-confirmation scout survived hard gates",
                }
            )
        for lane in required_lanes:
            if lane not in present_lanes and lane not in {"Scout"}:
                rows.append(
                    {
                        "Lane": lane,
                        "Status": "🔴 Blocked",
                        "Ticker": lane.upper().replace("/", "-"),
                        "Trade": f"No {lane} setup qualified",
                        "Expiry": "",
                        "Entry limit": "blocked",
                        "Live mid/natural": "",
                        "Max profit": "",
                        "Max loss": "",
                        "Target profit": "",
                        "Expected value source": "candidate audit",
                        "Edge sample size / win rate / avg P/L": "",
                        "Required confirmation": "strategy coverage, live pricing, edge, and risk gates must pass",
                        "Monitor trigger": "",
                        "Why Execute, Scout, Research, or Avoid": "lane did not produce a validated opportunity",
                    }
                )

    board = pd.DataFrame(rows)
    for col in OPPORTUNITY_COLUMNS:
        if col not in board.columns:
            board[col] = ""
    board = apply_lifecycle_triggers(board, asof=None)
    return board.head(active_cap) if active_cap else board


def opportunity_counts(board: pd.DataFrame) -> dict[str, int]:
    if board.empty:
        return {
            "execute": 0,
            "scout": 0,
            "momentum_debit": 0,
            "index_etf": 0,
            "portfolio_repair": 0,
            "wheel_cash": 0,
        }
    status = board["Status"].astype(str)
    lane = board["Lane"].astype(str)
    return {
        "execute": int(status.str.contains("Execute", regex=False).sum()),
        "scout": int(status.str.contains("Scout", regex=False).sum()),
        "momentum_debit": int(lane.eq("Momentum Debit").sum()),
        "index_etf": int(lane.eq("Index/ETF").sum()),
        "portfolio_repair": int(lane.eq("Portfolio Repair").sum()),
        "wheel_cash": int(lane.eq("Wheel/Cash").sum()),
    }


def classify_no_trade_audit(
    *,
    board: pd.DataFrame,
    scored: pd.DataFrame,
    data_quality: dict[str, Any] | None,
    portfolio: dict[str, Any] | None,
) -> dict[str, str]:
    counts = opportunity_counts(board)
    if counts["execute"] > 0:
        return {"classification": "execute_available", "exact_blocker": ""}
    blockers = list((data_quality or {}).get("critical_blockers") or [])
    if blockers:
        return {"classification": "data failure", "exact_blocker": ";".join(blockers)}
    if portfolio and portfolio.get("status") not in {None, "ok"}:
        return {"classification": "portfolio/risk constraint", "exact_blocker": _clean(portfolio.get("error") or portfolio.get("reason"))}
    if scored.empty:
        return {"classification": "missing strategy coverage", "exact_blocker": "candidate generation produced zero rows"}
    near_miss = pd.DataFrame()
    if not board.empty:
        status_text = board["Status"].astype(str)
        lane_text = board["Lane"].astype(str)
        near_miss = board[
            status_text.str.contains("Scout", regex=False)
            | (
                lane_text.isin(["Momentum Debit", "Index/ETF"])
                & ~status_text.str.contains("Blocked|Avoid", regex=True)
            )
        ]
    if not near_miss.empty:
        best = ""
        best = _clean(near_miss.iloc[0].get("Why Execute, Scout, Research, or Avoid"))
        return {
            "classification": "over-filtering",
            "exact_blocker": best or "near-miss lanes exist but at least one Execute guard remained unresolved",
        }
    text = ";".join(scored.get("hard_rejects", pd.Series("", index=scored.index)).fillna("").astype(str).tolist())
    penalties = ";".join(scored.get("penalties", pd.Series("", index=scored.index)).fillna("").astype(str).tolist())
    live_status = ";".join(scored.get("live_status", pd.Series("", index=scored.index)).fillna("").astype(str).tolist())
    combined = f"{text};{penalties};{live_status}".lower()
    live_passes = int(scored.get("live_status", pd.Series("", index=scored.index)).fillna("").astype(str).eq("PASS").sum())
    if ("chain_error" in combined or "live_unavailable" in combined) and live_passes == 0:
        return {"classification": "data failure", "exact_blocker": "Schwab live chain validation failed"}
    if "bid_ask" in combined or "liquidity" in combined or "natural" in combined:
        return {"classification": "execution/liquidity problem", "exact_blocker": "live quotes or liquidity failed execution quality"}
    if "risk" in combined or "portfolio" in combined:
        return {"classification": "portfolio/risk constraint", "exact_blocker": "risk cap or portfolio conflict blocked Execute"}
    return {"classification": "market quality problem", "exact_blocker": "no candidate had enough live edge, confirmation, and risk-adjusted quality"}


def write_recommendation_ledger(out_dir: Path, asof: dt.date, board: pd.DataFrame) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name
    ledger_rows: list[dict[str, Any]] = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for _, row in board.iterrows():
        strategy = _clean(row.get("strategy")) or _clean(row.get("Trade"))
        direction = _clean(row.get("direction"))
        recommended_limit = safe_float(row.get("recommended_limit"))
        ledger_rows.append(
            {
                "run_id": run_id,
                "generated_at_utc": now,
                "asof": str(asof),
                "report_date": str(asof),
                "lane": row.get("Lane"),
                "status": row.get("Status"),
                "ticker": row.get("Ticker"),
                "strategy": strategy,
                "setup_family": setup_family(strategy, direction),
                "direction": direction,
                "expiry": row.get("Expiry"),
                "recommended_limit": recommended_limit if math.isfinite(recommended_limit) else "",
                "actual_fill": row.get("actual_fill", ""),
                "close_fill": row.get("close_fill", ""),
                "realized_pnl": row.get("realized_pnl", ""),
                "mfe": row.get("mfe", ""),
                "mae": row.get("mae", ""),
                "thesis_worked": row.get("thesis_worked", ""),
                "reason_for_win_loss": row.get("reason_for_win_loss", ""),
                "slippage_vs_recommendation": row.get("slippage_vs_recommendation", ""),
                "monitor_triggered": row.get("monitor_triggered", ""),
                "monitor_trigger": row.get("Monitor trigger", ""),
                "source_report": str(out_dir / f"codexdaily_v3_report_{asof}.md"),
            }
        )
    run_ledger = pd.DataFrame(ledger_rows)
    run_path = out_dir / f"codexdaily_v3_recommendation_ledger_{asof}.csv"
    run_ledger.to_csv(run_path, index=False)
    global_path = out_dir.parent / "codexdaily_v3_recommendation_outcome_ledger.csv"
    if global_path.exists():
        try:
            existing = pd.read_csv(global_path)
            merged = pd.concat([existing, run_ledger], ignore_index=True)
            dedupe = ["run_id", "asof", "lane", "status", "ticker", "strategy", "expiry"]
            merged = merged.drop_duplicates(subset=[c for c in dedupe if c in merged.columns], keep="last")
        except Exception:
            merged = run_ledger
    else:
        merged = run_ledger
    merged.to_csv(global_path, index=False)
    return run_path, global_path
