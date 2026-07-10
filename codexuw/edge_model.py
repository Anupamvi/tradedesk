from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .credit_policy import assess_credit_spread
from .debit_policy import assess_debit_spread


CREDIT_DIRECTIONS = {"Bull Put", "Bear Call"}
BULLISH_DIRECTIONS = {"Bull Put", "Bull Call"}
BEARISH_DIRECTIONS = {"Bear Call", "Bear Put"}


EDGE_COLUMNS = [
    "edge_sample_size",
    "edge_win_rate",
    "edge_avg_pnl",
    "edge_profit_factor",
    "edge_max_drawdown",
    "edge_match_level",
    "edge_verdict",
    "edge_reason",
]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _direction_sign(direction: object) -> int:
    if str(direction or "") in BULLISH_DIRECTIONS:
        return 1
    if str(direction or "") in BEARISH_DIRECTIONS:
        return -1
    return 0


def _strategy_kind(row: pd.Series | dict[str, Any]) -> str:
    strategy = str(row.get("strategy", ""))
    direction = str(row.get("direction", ""))
    if direction in CREDIT_DIRECTIONS or "Credit" in strategy:
        return "Credit"
    return "Debit"


def _dte_bucket(value: object) -> str:
    dte = safe_float(value)
    if not math.isfinite(dte):
        return "unknown"
    if dte <= 10:
        return "07-10"
    if dte <= 21:
        return "11-21"
    if dte <= 35:
        return "22-35"
    return "36+"


def _premium_pct(row: pd.Series | dict[str, Any]) -> float:
    for key in [
        "credit_pct_width",
        "entry_credit_pct_width",
        "credit_pct_proxy",
        "debit_pct_width",
        "entry_debit_pct_width",
        "estimated_debit_pct_width",
        "estimated_eod_credit",
        "estimated_eod_debit",
    ]:
        value = safe_float(row.get(key))
        if math.isfinite(value) and value > 0:
            if key in {"estimated_eod_credit", "estimated_eod_debit"}:
                width = safe_float(row.get("preferred_width"), safe_float(row.get("spread_width")))
                return value / width if math.isfinite(width) and width > 0 else math.nan
            return value
    return math.nan


def _premium_bucket(row: pd.Series | dict[str, Any]) -> str:
    pct = _premium_pct(row)
    if not math.isfinite(pct):
        return "unknown"
    if pct < 0.16:
        return "lt16"
    if pct < 0.18:
        return "16-18"
    if pct < 0.25:
        return "18-25"
    if pct <= 0.30:
        return "25-30"
    return "30+"


def _expected_move_pct(row: pd.Series | dict[str, Any]) -> float:
    explicit = safe_float(row.get("expected_move_pct"))
    if math.isfinite(explicit) and explicit > 0:
        return explicit
    implied = safe_float(row.get("implied_move_perc"))
    if math.isfinite(implied) and implied > 0:
        return implied
    iv30d = safe_float(row.get("iv30d"))
    dte = safe_float(row.get("dte"))
    return iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan


def _distance_pct(row: pd.Series | dict[str, Any]) -> float:
    value = safe_float(row.get("distance_pct"))
    if math.isfinite(value):
        return value
    stock = safe_float(row.get("stock_price_live"), safe_float(row.get("stock_price_eod")))
    short = safe_float(row.get("short_strike"), safe_float(row.get("short_strike_eod")))
    direction = str(row.get("direction", ""))
    if not math.isfinite(stock) or stock <= 0 or not math.isfinite(short):
        return math.nan
    if direction == "Bull Put":
        return (stock - short) / stock
    if direction == "Bear Call":
        return (short - stock) / stock
    return safe_float(row.get("breakeven_distance_pct"))


def _expected_ratio(row: pd.Series | dict[str, Any]) -> float:
    explicit = safe_float(row.get("expected_move_ratio"))
    if math.isfinite(explicit) and explicit > 0:
        return explicit
    expected = _expected_move_pct(row)
    if not math.isfinite(expected) or expected <= 0:
        return math.nan
    if _strategy_kind(row) == "Debit":
        breakeven_distance = safe_float(row.get("breakeven_distance_pct"))
        return expected / max(breakeven_distance, 0.001) if math.isfinite(breakeven_distance) else math.nan
    distance = _distance_pct(row)
    return distance / expected if math.isfinite(distance) else math.nan


def _ratio_bucket(row: pd.Series | dict[str, Any]) -> str:
    ratio = _expected_ratio(row)
    if not math.isfinite(ratio):
        return "unknown"
    if ratio < 0.55:
        return "lt55"
    if ratio < 0.65:
        return "55-65"
    if ratio < 1.0:
        return "65-100"
    return "100+"


def _flow_align_bucket(row: pd.Series | dict[str, Any]) -> str:
    align = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0)) * _direction_sign(row.get("direction"))
    if not math.isfinite(align):
        return "unknown"
    if align < 0:
        return "contrary"
    if align < 0.04:
        return "weak"
    if align < 0.10:
        return "moderate"
    return "strong"


def _iv_bucket(row: pd.Series | dict[str, Any]) -> str:
    iv_rank = safe_float(row.get("iv_rank"))
    if math.isfinite(iv_rank):
        if iv_rank < 25:
            return "low"
        if iv_rank < 60:
            return "medium"
        return "high"
    iv30d = safe_float(row.get("iv30d"))
    if math.isfinite(iv30d):
        if iv30d < 0.25:
            return "low"
        if iv30d < 0.55:
            return "medium"
        return "high"
    return "unknown"


def _iv_hv_bucket(row: pd.Series | dict[str, Any]) -> str:
    ratio = safe_float(row.get("iv_hv_ratio"))
    if not math.isfinite(ratio) or ratio <= 0:
        implied = safe_float(row.get("iv30d"))
        realized = safe_float(row.get("realized_volatility_30d"), safe_float(row.get("volatility")))
        ratio = implied / realized if math.isfinite(implied) and math.isfinite(realized) and realized > 0 else math.nan
    if not math.isfinite(ratio):
        return "unknown"
    if ratio < 0.90:
        return "lt0.90"
    if ratio < 1.00:
        return "0.90-1.00"
    return "1.00+"


def _quote_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = safe_float(row.get("quote_width_pct"), safe_float(row.get("entry_quote_width_pct")))
    if not math.isfinite(value):
        return "unknown"
    if value <= 0.20:
        return "clean"
    if value <= 0.35:
        return "usable"
    if value <= 0.65:
        return "marginal"
    return "wide"


def _earnings_bucket(row: pd.Series | dict[str, Any]) -> str:
    value = row.get("next_earnings_dt")
    asof = row.get("asof")
    try:
        earnings = pd.to_datetime(value, errors="coerce")
        day = pd.to_datetime(asof, errors="coerce")
    except Exception:
        return "unknown"
    if pd.isna(earnings) or pd.isna(day):
        return "unknown"
    distance = int((earnings.date() - day.date()).days)
    if distance < 0:
        return "past"
    if distance <= 7:
        return "0-7"
    if distance <= 14:
        return "8-14"
    return "15+"


def _feature_record(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    direction = str(row.get("direction", ""))
    trend = str(row.get("regime_trend", row.get("regime", row.get("trend", ""))) or "")
    return {
        "ticker": str(row.get("ticker", "")).upper(),
        "sector": str(row.get("sector", "") or ""),
        "direction": direction,
        "strategy_kind": _strategy_kind(row),
        "dte_bucket": _dte_bucket(row.get("dte")),
        "premium_bucket": _premium_bucket(row),
        "expected_move_bucket": _ratio_bucket(row),
        "flow_align_bucket": _flow_align_bucket(row),
        "flow_quality": str(row.get("flow_quality", "") or ""),
        "oi_carryover_status": str(row.get("oi_carryover_status", "") or ""),
        "regime_trend": trend,
        "iv_bucket": _iv_bucket(row),
        "iv_hv_bucket": _iv_hv_bucket(row),
        "quote_quality_bucket": _quote_bucket(row),
        "earnings_bucket": _earnings_bucket(row),
    }


def _max_drawdown(values: pd.Series) -> float:
    total = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in pd.to_numeric(values, errors="coerce").fillna(0.0):
        total += float(value)
        peak = max(peak, total)
        drawdown = min(drawdown, total - peak)
    return drawdown


def _metrics(part: pd.DataFrame) -> dict[str, Any]:
    if part.empty:
        return {
            "edge_sample_size": 0,
            "edge_win_rate": math.nan,
            "edge_avg_pnl": math.nan,
            "edge_profit_factor": math.nan,
            "edge_max_drawdown": math.nan,
        }
    pnl = pd.to_numeric(part["pnl_1x"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    return {
        "edge_sample_size": int(len(part)),
        "edge_win_rate": float(part["exact_win"].astype(str).str.lower().eq("true").mean()),
        "edge_avg_pnl": float(pnl.mean()),
        "edge_profit_factor": float(wins / losses) if losses > 0 else (math.inf if wins > 0 else math.nan),
        "edge_max_drawdown": float(_max_drawdown(pnl)),
    }


def _verdict(metrics: dict[str, Any]) -> str:
    sample = int(metrics.get("edge_sample_size") or 0)
    win_rate = safe_float(metrics.get("edge_win_rate"))
    avg_pnl = safe_float(metrics.get("edge_avg_pnl"))
    profit_factor = safe_float(metrics.get("edge_profit_factor"))
    if sample <= 0:
        return "unavailable"
    if sample < 3:
        return "thin_sample"
    if math.isfinite(avg_pnl) and avg_pnl < 0 and (not math.isfinite(win_rate) or win_rate < 0.50):
        return "negative"
    if math.isfinite(profit_factor) and profit_factor < 0.80 and math.isfinite(win_rate) and win_rate < 0.50:
        return "negative"
    if math.isfinite(avg_pnl) and avg_pnl > 0 and math.isfinite(win_rate) and win_rate >= 0.60 and (
        not math.isfinite(profit_factor) or profit_factor >= 1.20
    ):
        return "positive"
    if math.isfinite(avg_pnl) and avg_pnl >= 0 and (not math.isfinite(win_rate) or win_rate >= 0.50):
        return "acceptable"
    return "thin_sample"


def _empty_edge(reason: str = "no replay detail matched candidate pattern") -> dict[str, Any]:
    return {
        "edge_sample_size": 0,
        "edge_win_rate": math.nan,
        "edge_avg_pnl": math.nan,
        "edge_profit_factor": math.nan,
        "edge_max_drawdown": math.nan,
        "edge_match_level": "unavailable",
        "edge_verdict": "unavailable",
        "edge_reason": reason,
    }


def load_replay_edge_history(
    out_root: Path,
    *,
    asof: object | None = None,
    history_namespace: str | None = None,
) -> pd.DataFrame:
    if history_namespace:
        roots = [
            child
            for child in out_root.iterdir()
            if child.is_dir() and child.name.startswith(history_namespace)
        ] if out_root.exists() else []
        if out_root.is_dir() and out_root.name.startswith(history_namespace):
            roots.append(out_root)
        paths = sorted(
            {path for root in roots for path in root.rglob("codexuw_replay_detail.csv")},
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    else:
        paths = sorted(out_root.rglob("codexuw_replay_detail.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "exact_evaluated" not in df.columns or "pnl_1x" not in df.columns:
            continue
        df = df[df["exact_evaluated"].map(_truthy)].copy()
        df = df[pd.to_numeric(df["pnl_1x"], errors="coerce").notna()].copy()
        if asof is not None:
            cutoff = pd.to_datetime(asof, errors="coerce")
            if not pd.isna(cutoff):
                source_day = pd.to_datetime(df.get("asof"), errors="coerce")
                exit_day = pd.to_datetime(df.get("exit_day"), errors="coerce")
                df = df[source_day.lt(cutoff) & exit_day.lt(cutoff)].copy()
        if df.empty:
            continue
        df["edge_source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    key_cols = [
        "asof",
        "ticker",
        "direction",
        "expiry",
        "short_strike_eod",
        "long_strike_eod",
        "entry_credit_pct_width",
        "entry_debit_pct_width",
    ]
    history = history.drop_duplicates(subset=[c for c in key_cols if c in history.columns], keep="first")
    features = pd.DataFrame([_feature_record(row) for _, row in history.iterrows()])
    for col in features.columns:
        history[col] = features[col].values
    return history


def match_replay_edge(row: pd.Series | dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return _empty_edge("no replay detail files with exact P/L were found")
    feat = _feature_record(row)
    df = history.copy()
    if feat["strategy_kind"] == "Credit":
        candidate_policy_ok, _ = assess_credit_spread(row, live=False)
        if not candidate_policy_ok:
            return _empty_edge("candidate does not meet the accepted distance-qualified credit policy")
        selected = df.get("decision_pass", pd.Series(False, index=df.index)).map(_truthy)
        policy_ok = df.apply(lambda item: assess_credit_spread(item, live=False)[0], axis=1)
        df = df[selected & policy_ok].copy()
        if df.empty:
            return _empty_edge("no decision-selected distance-qualified credit replay history matched candidate")
    if feat["strategy_kind"] == "Debit" and "bot_flow_source_status" in df.columns:
        full_bot = df["bot_flow_source_status"].astype(str).eq("bot_eod_loaded")
        directional_contract = df.get("flow_quality", pd.Series("", index=df.index)).astype(str).eq("directional")
        df = df[full_bot | directional_contract].copy()
        if df.empty:
            return _empty_edge("no side-aware bot or directional-contract debit replay history matched candidate")

    debit_policy_mask = pd.Series(False, index=df.index)
    if feat["strategy_kind"] == "Debit":
        candidate_policy_ok, _ = assess_debit_spread(row, live=False)
        if candidate_policy_ok:
            debit_policy_mask = df.apply(lambda item: assess_debit_spread(item, live=False)[0], axis=1)

    match_specs = [
        (
            "exact",
            (df["ticker"].eq(feat["ticker"]))
            & (df["direction"].eq(feat["direction"]))
            & (df["dte_bucket"].eq(feat["dte_bucket"]))
            & (df["premium_bucket"].eq(feat["premium_bucket"]))
            & (df["expected_move_bucket"].eq(feat["expected_move_bucket"]))
            & (df["regime_trend"].eq(feat["regime_trend"]))
            & (df["iv_hv_bucket"].eq(feat["iv_hv_bucket"])),
        ),
        ("ticker_direction", (df["ticker"].eq(feat["ticker"])) & (df["direction"].eq(feat["direction"]))),
        (
            "strategy_regime",
            (df["direction"].eq(feat["direction"]))
            & (df["strategy_kind"].eq(feat["strategy_kind"]))
            & (df["regime_trend"].eq(feat["regime_trend"]))
            & (df["dte_bucket"].eq(feat["dte_bucket"]))
            & (df["premium_bucket"].eq(feat["premium_bucket"])),
        ),
        (
            "debit_policy_sleeve",
            debit_policy_mask
            & (df["direction"].eq(feat["direction"]))
            & (df["strategy_kind"].eq("Debit")),
        ),
        (
            "credit_policy_sleeve",
            (df["direction"].eq(feat["direction"]))
            & (df["strategy_kind"].eq("Credit")),
        ),
        (
            "broad_pattern",
            (df["direction"].eq(feat["direction"]))
            & (df["strategy_kind"].eq(feat["strategy_kind"]))
            & (df["dte_bucket"].eq(feat["dte_bucket"]))
            & (df["premium_bucket"].eq(feat["premium_bucket"])),
        ),
        (
            "broad_pattern",
            (df["direction"].eq(feat["direction"]))
            & (df["strategy_kind"].eq(feat["strategy_kind"]))
            & (df["expected_move_bucket"].eq(feat["expected_move_bucket"])),
        ),
    ]
    best_level = "unavailable"
    best = pd.DataFrame()
    minimum_sample = 8
    for level, mask in match_specs:
        if feat["strategy_kind"] == "Debit" and level == "broad_pattern":
            continue
        part = df[mask].copy()
        if part.empty:
            continue
        best_level = level
        best = part
        if len(part) >= minimum_sample:
            break
    if best.empty:
        return _empty_edge(
            f"no replay match for {feat['ticker']} {feat['direction']} {feat['dte_bucket']} {feat['premium_bucket']}"
        )

    metrics = _metrics(best)
    verdict = "thin_sample" if metrics["edge_sample_size"] < minimum_sample else _verdict(metrics)
    sample = metrics["edge_sample_size"]
    win_rate = safe_float(metrics["edge_win_rate"])
    avg_pnl = safe_float(metrics["edge_avg_pnl"])
    reason = (
        f"{best_level} replay match: {feat['direction']} {feat['strategy_kind']} "
        f"{feat['dte_bucket']} DTE, {feat['premium_bucket']} premium, "
        f"{feat['expected_move_bucket']} expected-move bucket, "
        f"{feat['iv_hv_bucket']} IV/HV bucket; sample {sample}, "
        f"win {win_rate:.1%}, avg P/L ${avg_pnl:.2f}"
        if math.isfinite(win_rate) and math.isfinite(avg_pnl)
        else f"{best_level} replay match with sample {sample}"
    )
    return {
        **metrics,
        "edge_match_level": best_level,
        "edge_verdict": verdict,
        "edge_reason": reason,
    }


def apply_replay_edge_model(
    scored: pd.DataFrame,
    out_root: Path,
    *,
    asof: object | None = None,
    history_namespace: str | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    history = load_replay_edge_history(out_root, asof=asof, history_namespace=history_namespace)
    out = scored.copy()
    out["edge_history_namespace"] = history_namespace or "legacy_all_replays"
    out["edge_history_cutoff"] = str(asof or "")
    out["edge_history_rows"] = int(len(history))
    for col in EDGE_COLUMNS:
        if col not in out.columns:
            out[col] = math.nan if col not in {"edge_match_level", "edge_verdict", "edge_reason"} else ""
    for idx, row in out.iterrows():
        edge = match_replay_edge(row, history)
        for key, value in edge.items():
            out.at[idx, key] = value
        verdict = str(edge.get("edge_verdict") or "")
        existing = str(row.get("replay_ev_verdict") or "")
        if verdict in {"positive", "acceptable"}:
            out.at[idx, "replay_ev_verdict"] = verdict
            out.at[idx, "replay_pattern"] = str(edge.get("edge_match_level") or "")
        elif verdict == "negative":
            out.at[idx, "replay_ev_verdict"] = "negative"
            hard = str(row.get("hard_rejects") or "")
            tokens = [token for token in hard.split(";") if token]
            if "negative_replay_edge" not in tokens:
                tokens.append("negative_replay_edge")
            out.at[idx, "hard_rejects"] = ";".join(tokens)
        elif verdict == "thin_sample":
            out.at[idx, "replay_ev_verdict"] = "thin_sample"
        elif existing == "acceptable_proxy":
            out.at[idx, "replay_ev_verdict"] = "acceptable_proxy"
        else:
            out.at[idx, "replay_ev_verdict"] = "unavailable"
    return out
