from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


SHADOW_POLICY_VERSION = "debit-shadow-v1-bull-call-dte28-45"
ALLOWED_DIRECTION = "Bull Call"
ALLOWED_REGIME = "uptrend"
MIN_DTE = 28
MAX_DTE = 45
MAX_DEBIT_PCT_WIDTH = 0.45
MIN_REWARD_RISK = 1.25
MIN_FLOW_ALIGN = 0.20
MAX_IV_RANK = 55.0
MAX_QUOTE_WIDTH_PCT = 0.35
SHADOW_SPLIT_DAY = dt.date(2026, 6, 15)
WHY_NOT_EXECUTE = "debit_shadow_only; never Execute; never decision_pass"

LEDGER_COLUMNS = [
    "shadow_policy_version",
    "asof",
    "ticker",
    "direction",
    "strategy",
    "sell_leg",
    "buy_leg",
    "short_strike",
    "long_strike",
    "expiry",
    "dte",
    "regime",
    "debit",
    "debit_pct",
    "width",
    "reward_risk",
    "flow_align",
    "iv_rank",
    "quote_width_pct",
    "mid",
    "gate_bull_call",
    "gate_uptrend",
    "gate_dte_28_45",
    "gate_debit_le_45pct",
    "gate_reward_risk_ge_1_25",
    "gate_flow_align_ge_0_20",
    "gate_iv_rank_le_55",
    "gate_quote_width_le_35pct",
    "shadow_qualified",
    "fail_reasons",
    "why_not_execute",
    "execution_authorized",
    "decision_pass",
    "exact_evaluated",
    "pnl_1x",
]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _asof(value: object) -> str:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    text = _clean(value)
    return text[:10] if text else ""


def _pick(row: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(row, "get"):
            value = row.get(name)
        else:
            value = row[name] if name in row else None
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        if _clean(value) == "" or _clean(value).lower() == "nan":
            continue
        return value
    return default


def _flow_align(row: Any) -> float:
    bias = safe_float(_pick(row, ("combined_flow_bias", "flow_bias"), math.nan))
    direction = _clean(_pick(row, ("direction",), ""))
    sign = 1.0 if direction == "Bull Call" else -1.0 if direction == "Bear Put" else 0.0
    return bias * sign if sign and math.isfinite(bias) else math.nan


def _debit(row: Any) -> float:
    return safe_float(
        _pick(row, ("entry_debit", "debit", "estimated_eod_debit", "live_debit"), math.nan)
    )


def _width(row: Any) -> float:
    width = safe_float(_pick(row, ("entry_width", "preferred_width", "spread_width"), math.nan))
    if math.isfinite(width) and width > 0:
        return width
    short = safe_float(_pick(row, ("short_strike_eod", "short_strike"), math.nan))
    long = safe_float(_pick(row, ("long_strike_eod", "long_strike"), math.nan))
    if math.isfinite(short) and math.isfinite(long):
        return abs(long - short)
    return math.nan


def _debit_pct(row: Any) -> float:
    pct = safe_float(_pick(row, ("entry_debit_pct_width", "debit_pct_width", "debit_pct_proxy"), math.nan))
    if math.isfinite(pct):
        return pct
    debit = _debit(row)
    width = _width(row)
    return debit / width if math.isfinite(debit) and math.isfinite(width) and width > 0 else math.nan


def _reward_risk(row: Any) -> float:
    rr = safe_float(_pick(row, ("reward_risk",), math.nan))
    if math.isfinite(rr):
        return rr
    debit = _debit(row)
    width = _width(row)
    if math.isfinite(debit) and debit > 0 and math.isfinite(width) and width > debit:
        return (width - debit) / debit
    return math.nan


def _dte(row: Any) -> float:
    return safe_float(_pick(row, ("entry_dte", "dte"), math.nan))


def _quote_width(row: Any) -> float:
    return safe_float(_pick(row, ("entry_quote_width_pct", "quote_width_pct"), math.nan))


def _mid(row: Any) -> float:
    mid = safe_float(_pick(row, ("entry_mid", "entry_mid_debit", "mid"), math.nan))
    if math.isfinite(mid):
        return mid
    debit = _debit(row)
    return debit if math.isfinite(debit) else math.nan


def _is_debit_candidate(row: Any) -> bool:
    direction = _clean(_pick(row, ("direction",), ""))
    strategy = _clean(_pick(row, ("strategy", "strategy_kind"), ""))
    return direction in {"Bull Call", "Bear Put"} or "Debit" in strategy


def assess_shadow_gates(row: Any) -> dict[str, Any]:
    direction = _clean(_pick(row, ("direction",), ""))
    regime = _clean(_pick(row, ("regime", "regime_trend", "trend"), "")).lower()
    dte = _dte(row)
    debit_pct = _debit_pct(row)
    reward_risk = _reward_risk(row)
    flow = _flow_align(row)
    iv_rank = safe_float(_pick(row, ("iv_rank",), math.nan))
    quote_width = _quote_width(row)
    gates = {
        "gate_bull_call": direction == ALLOWED_DIRECTION,
        "gate_uptrend": regime == ALLOWED_REGIME,
        "gate_dte_28_45": math.isfinite(dte) and MIN_DTE <= dte <= MAX_DTE,
        "gate_debit_le_45pct": math.isfinite(debit_pct) and 0 < debit_pct <= MAX_DEBIT_PCT_WIDTH,
        "gate_reward_risk_ge_1_25": math.isfinite(reward_risk) and reward_risk >= MIN_REWARD_RISK,
        "gate_flow_align_ge_0_20": math.isfinite(flow) and flow >= MIN_FLOW_ALIGN,
        "gate_iv_rank_le_55": math.isfinite(iv_rank) and iv_rank <= MAX_IV_RANK,
        "gate_quote_width_le_35pct": math.isfinite(quote_width) and quote_width <= MAX_QUOTE_WIDTH_PCT,
    }
    fails: list[str] = []
    if not gates["gate_bull_call"]:
        fails.append(f"direction_not_bull_call:{direction or 'unknown'}")
    if not gates["gate_uptrend"]:
        fails.append(f"regime_not_uptrend:{regime or 'unknown'}")
    if not gates["gate_dte_28_45"]:
        fails.append(f"dte_outside_{MIN_DTE}_{MAX_DTE}")
    if not gates["gate_debit_le_45pct"]:
        fails.append("debit_pct_width_above_0.45")
    if not gates["gate_reward_risk_ge_1_25"]:
        fails.append("reward_risk_below_1.25")
    if not gates["gate_flow_align_ge_0_20"]:
        fails.append("flow_alignment_below_0.20")
    if not gates["gate_iv_rank_le_55"]:
        fails.append("iv_rank_above_55")
    if not gates["gate_quote_width_le_35pct"]:
        fails.append("quote_width_above_0.35")
    qualified = all(gates.values())
    why = WHY_NOT_EXECUTE
    if fails:
        why = WHY_NOT_EXECUTE + "; " + "|".join(fails)
    return {
        **gates,
        "shadow_qualified": qualified,
        "fail_reasons": "|".join(fails),
        "why_not_execute": why,
        "dte": dte,
        "regime": regime,
        "debit": _debit(row),
        "debit_pct": debit_pct,
        "width": _width(row),
        "reward_risk": reward_risk,
        "flow_align": flow,
        "iv_rank": iv_rank,
        "quote_width_pct": quote_width,
        "mid": _mid(row),
        "direction": direction,
    }


def build_shadow_rows(frame: pd.DataFrame, *, asof: dt.date | str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, raw in frame.iterrows():
        if not _is_debit_candidate(raw):
            continue
        assessed = assess_shadow_gates(raw)
        day = _asof(asof if asof is not None else raw.get("asof"))
        rows.append(
            {
                "shadow_policy_version": SHADOW_POLICY_VERSION,
                "asof": day,
                "ticker": _clean(_pick(raw, ("ticker",), "")).upper(),
                "direction": assessed["direction"],
                "strategy": _clean(_pick(raw, ("strategy",), "")),
                "sell_leg": _clean(_pick(raw, ("short_leg_eod", "short_leg", "sell_leg"), "")),
                "buy_leg": _clean(_pick(raw, ("long_leg_eod", "long_leg", "buy_leg"), "")),
                "short_strike": safe_float(_pick(raw, ("short_strike_eod", "short_strike"), math.nan)),
                "long_strike": safe_float(_pick(raw, ("long_strike_eod", "long_strike"), math.nan)),
                "expiry": _asof(_pick(raw, ("expiry",), "")),
                "dte": assessed["dte"],
                "regime": assessed["regime"],
                "debit": assessed["debit"],
                "debit_pct": assessed["debit_pct"],
                "width": assessed["width"],
                "reward_risk": assessed["reward_risk"],
                "flow_align": assessed["flow_align"],
                "iv_rank": assessed["iv_rank"],
                "quote_width_pct": assessed["quote_width_pct"],
                "mid": assessed["mid"],
                "gate_bull_call": assessed["gate_bull_call"],
                "gate_uptrend": assessed["gate_uptrend"],
                "gate_dte_28_45": assessed["gate_dte_28_45"],
                "gate_debit_le_45pct": assessed["gate_debit_le_45pct"],
                "gate_reward_risk_ge_1_25": assessed["gate_reward_risk_ge_1_25"],
                "gate_flow_align_ge_0_20": assessed["gate_flow_align_ge_0_20"],
                "gate_iv_rank_le_55": assessed["gate_iv_rank_le_55"],
                "gate_quote_width_le_35pct": assessed["gate_quote_width_le_35pct"],
                "shadow_qualified": assessed["shadow_qualified"],
                "fail_reasons": assessed["fail_reasons"],
                "why_not_execute": assessed["why_not_execute"],
                "execution_authorized": False,
                "decision_pass": False,
                "exact_evaluated": bool(raw.get("exact_evaluated")) if "exact_evaluated" in raw.index else False,
                "pnl_1x": safe_float(raw.get("pnl_1x"), math.nan) if "pnl_1x" in raw.index else math.nan,
            }
        )
    out = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if not out.empty:
        out["decision_pass"] = False
        out["execution_authorized"] = False
    return out


def append_shadow_ledger(path: Path, incoming: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = incoming.copy()
    if incoming.empty:
        if not path.exists():
            pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(path, index=False)
        return path
    incoming["decision_pass"] = False
    incoming["execution_authorized"] = False
    if path.exists():
        existing = pd.read_csv(path, low_memory=False)
        days = set(incoming["asof"].astype(str))
        keep = existing[~existing["asof"].astype(str).isin(days)] if "asof" in existing.columns else existing
        combined = pd.concat([keep, incoming], ignore_index=True)
    else:
        combined = incoming
    for col in LEDGER_COLUMNS:
        if col not in combined.columns:
            combined[col] = False if col in {"decision_pass", "execution_authorized", "shadow_qualified"} else ""
    combined = combined[LEDGER_COLUMNS]
    combined.to_csv(path, index=False)
    return path


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"n": 0, "evaluated": 0, "win_rate": None, "profit_factor": None, "total_pnl_1x": 0.0}
    work = frame.copy()
    pnl = pd.to_numeric(work.get("pnl_1x"), errors="coerce")
    evaluated = work
    if "exact_evaluated" in work.columns:
        evaluated = work[work["exact_evaluated"].map(lambda v: str(v).strip().lower() in {"true", "1"})]
        pnl = pd.to_numeric(evaluated.get("pnl_1x"), errors="coerce")
    pnl = pnl.dropna()
    wins = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    losses = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    pf = wins / losses if losses > 0 else (math.inf if wins > 0 else None)
    return {
        "n": int(len(frame)),
        "evaluated": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else None,
        "profit_factor": pf,
        "total_pnl_1x": float(pnl.sum()) if not pnl.empty else 0.0,
        "avg_pnl_1x": float(pnl.mean()) if not pnl.empty else None,
    }


def split_shadow_metrics(ledger: pd.DataFrame, split_day: dt.date = SHADOW_SPLIT_DAY) -> dict[str, Any]:
    qualified = ledger[ledger.get("shadow_qualified", pd.Series(dtype=bool)).map(lambda v: str(v).strip().lower() in {"true", "1"})].copy() if not ledger.empty else ledger
    if qualified.empty:
        empty = _metrics(qualified)
        return {"split_day": split_day.isoformat(), "all": empty, "train": empty, "test": empty}
    asof = pd.to_datetime(qualified["asof"], errors="coerce")
    train = qualified[asof <= pd.Timestamp(split_day)]
    test = qualified[asof > pd.Timestamp(split_day)]
    return {
        "split_day": split_day.isoformat(),
        "policy_version": SHADOW_POLICY_VERSION,
        "all": _metrics(qualified),
        "train": _metrics(train),
        "test": _metrics(test),
        "ledger_rows": int(len(ledger)),
        "qualified_rows": int(len(qualified)),
    }


def format_shadow_report_section(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Debit Shadow Train/Test",
        "",
        f"- Policy: `{metrics.get('policy_version', SHADOW_POLICY_VERSION)}`",
        f"- Frozen spec: Bull Call, uptrend, DTE {MIN_DTE}–{MAX_DTE}, debit ≤ {MAX_DEBIT_PCT_WIDTH:.0%} width, R/R ≥ {MIN_REWARD_RISK}, flow ≥ {MIN_FLOW_ALIGN:.2f}, IV rank ≤ {int(MAX_IV_RANK)}, quote width ≤ {MAX_QUOTE_WIDTH_PCT:.0%}",
        "- Shadow only: never `decision_pass`, never Execute, not mixed into credit selected.",
        f"- Split day: {metrics.get('split_day', SHADOW_SPLIT_DAY.isoformat())}",
        f"- Ledger rows (pass and fail): {metrics.get('ledger_rows', 0)}",
        f"- Qualified (all gates pass): {metrics.get('qualified_rows', 0)}",
        "",
    ]
    table_rows = []
    for name in ["train", "test", "all"]:
        item = metrics.get(name, {})
        win = item.get("win_rate")
        pf = item.get("profit_factor")
        table_rows.append(
            {
                "Split": name,
                "n": item.get("evaluated", 0),
                "win": f"{win:.1%}" if win is not None else "",
                "PF": f"{pf:.3f}" if isinstance(pf, float) and math.isfinite(pf) else ("inf" if pf == math.inf else ""),
                "$": f"${item.get('total_pnl_1x', 0.0):,.2f}",
            }
        )
    lines.append(pd.DataFrame(table_rows).to_markdown(index=False))
    lines.append("")
    return lines


def write_daily_debit_shadow(*, scored: pd.DataFrame, asof: dt.date, out_dir: Path) -> Path:
    rows = build_shadow_rows(scored, asof=asof)
    run_copy = Path(out_dir) / f"debit_shadow_{asof}.csv"
    rows.to_csv(run_copy, index=False)
    ledger = Path(out_dir).parent / "debit_shadow" / "debit_shadow_ledger.csv"
    append_shadow_ledger(ledger, rows)
    return ledger


def write_replay_debit_shadow(
    *,
    detail: pd.DataFrame,
    out_dir: Path,
    split_day: dt.date = SHADOW_SPLIT_DAY,
) -> tuple[dict[str, Path], dict[str, Any]]:
    rows = build_shadow_rows(detail)
    shadow_dir = Path(out_dir) / "debit_shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = shadow_dir / "debit_shadow_ledger.csv"
    append_shadow_ledger(ledger_path, rows)
    metrics = split_shadow_metrics(rows, split_day=split_day)
    metrics_path = shadow_dir / "debit_shadow_metrics.json"
    def _jsonable(value: Any) -> Any:
        if value is math.inf:
            return "inf"
        if isinstance(value, dict):
            return {key: _jsonable(item) for key, item in value.items()}
        return value

    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_path = shadow_dir / "debit_shadow_report.md"
    report_path.write_text("\n".join(["# Debit Shadow"] + format_shadow_report_section(metrics)), encoding="utf-8")
    return {"ledger": ledger_path, "metrics": metrics_path, "report": report_path}, metrics
