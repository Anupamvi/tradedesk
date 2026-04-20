#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from uwos import swing_trend_pipeline as swing
from uwos import trend_quote_replay


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_LOOKBACK = 90
DEFAULT_TOP = 15
DEFAULT_CANDIDATE_TOP = 10
DEFAULT_REPO_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
DEFAULT_MAX_BID_ASK_TO_PRICE_PCT = 0.30
DEFAULT_MAX_BID_ASK_TO_WIDTH_PCT = 0.10
DEFAULT_MAX_SHORT_DELTA = 0.30
DEFAULT_CANDIDATE_MIN_SCORE = 60.0
DEFAULT_CANDIDATE_MIN_CONFIRMATIONS = 4
DEFAULT_CANDIDATE_MAX_CONFLICTS = 1
DEFAULT_MIN_UNDERLYING_PRICE = 20.0
DEFAULT_MIN_DEBIT_SPREAD_PRICE = 0.75
DEFAULT_MIN_WHALE_APPEARANCES = 8
DEFAULT_MIN_DIRECTIONAL_FLOW_SCORE = 60.0
DEFAULT_MIN_DIRECTIONAL_PRICE_SCORE = 60.0
DEFAULT_MAX_DEBIT_TO_WIDTH_PCT = 0.50
DEFAULT_MAX_LONG_STRIKE_OTM_PCT = 0.02
DEFAULT_MIN_WORKUP_SIGNALS = 50
DEFAULT_MAX_CONVICTION_MIN_SCORE = 75.0
DEFAULT_MAX_CONVICTION_MIN_EDGE = 15.0
DEFAULT_MAX_CONVICTION_MIN_SIGNALS = 100
DEFAULT_MAX_CONVICTION_MIN_WHALE_COVERAGE = 0.60
DEFAULT_WALK_FORWARD_SAMPLES = 60
DEFAULT_WALK_FORWARD_HORIZONS = "5,10,20"
DEFAULT_WALK_FORWARD_TOP = 3
RESEARCH_AUDIT_MIN_OUTCOMES = 20
RESEARCH_AUDIT_MIN_UNIQUE_SETUPS = 20
FAMILY_AUDIT_MIN_TRAIN_SETUPS = 20
FAMILY_AUDIT_MIN_VALIDATION_SETUPS = 8
FAMILY_AUDIT_MIN_VALIDATION_DATES = 4
FAMILY_AUDIT_MIN_PROFIT_FACTOR = 1.20
TICKER_PLAYBOOK_MIN_TRAIN_SETUPS = 8
TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS = 5
TICKER_PLAYBOOK_MIN_VALIDATION_DATES = 3
TICKER_PLAYBOOK_MIN_PROFIT_FACTOR = 1.20
TICKER_PLAYBOOK_MIN_VALIDATION_HIT = 0.55
ROLLING_PLAYBOOK_MIN_FORWARD_TESTS = 3
ROLLING_PLAYBOOK_MIN_FORWARD_DATES = 2
ROLLING_PLAYBOOK_MIN_PROFIT_FACTOR = 1.05
ROLLING_PLAYBOOK_MIN_HIT_RATE = 0.50
DEFAULT_TRACKING_FILE_NAME = "trend-analysis-trade-tracker.csv"
WALK_FORWARD_COLUMNS = [
    "signal_date",
    "horizon_market_days",
    "exit_date",
    "selected_rank",
    "ticker",
    "direction",
    "strategy",
    "trade_setup",
    "swing_score",
    "edge_pct",
    "backtest_signals",
    "backtest_verdict",
    "entry_available",
    "entry_net",
    "exit_net",
    "pnl",
    "return_on_risk",
    "win",
    "outcome_verdict",
    "outcome_status",
    "outcome_reason",
]
RESEARCH_AUDIT_COLUMNS = [
    "policy",
    "outcomes",
    "unique_setups",
    "avg_horizons_per_setup",
    "hit_rate",
    "avg_pnl",
    "median_pnl",
    "avg_return_on_risk",
    "worst_pnl",
    "verdict",
]
RESEARCH_OUTCOME_COLUMNS = [
    "policy",
    "signal_date",
    "horizon_market_days",
    "ticker",
    "direction",
    "strategy",
    "trade_setup",
    "swing_score",
    "edge_pct",
    "backtest_signals",
    "backtest_verdict",
    "price_direction",
    "price_trend",
    "flow_direction",
    "flow_persistence",
    "oi_direction",
    "oi_momentum",
    "dp_direction",
    "dp_confirmation",
    "sector",
    "whale_appearances",
    "days_observed",
    "latest_close",
    "latest_iv_rank",
    "iv_level",
    "cost_type",
    "track",
    "variant_tag",
    "target_dte",
    "base_gate_reasons",
    "quality_reject_reasons",
    "entry_net",
    "exit_net",
    "pnl",
    "return_on_risk",
]
RESEARCH_HORIZON_AUDIT_COLUMNS = [
    "policy",
    "horizon_market_days",
    "outcomes",
    "unique_setups",
    "hit_rate",
    "avg_pnl",
    "median_pnl",
    "avg_return_on_risk",
    "worst_pnl",
    "verdict",
]
STRATEGY_FAMILY_AUDIT_COLUMNS = [
    "family",
    "horizon_market_days",
    "description",
    "overall_outcomes",
    "overall_unique_setups",
    "overall_dates",
    "overall_hit_rate",
    "overall_avg_pnl",
    "overall_profit_factor",
    "overall_max_drawdown",
    "train_outcomes",
    "train_unique_setups",
    "train_dates",
    "train_hit_rate",
    "train_avg_pnl",
    "train_profit_factor",
    "train_max_drawdown",
    "validation_outcomes",
    "validation_unique_setups",
    "validation_dates",
    "validation_hit_rate",
    "validation_avg_pnl",
    "validation_profit_factor",
    "validation_max_drawdown",
    "worst_pnl",
    "verdict",
]
TICKER_PLAYBOOK_AUDIT_COLUMNS = [
    "ticker",
    "direction",
    "strategy",
    "horizon_market_days",
    "description",
    "overall_outcomes",
    "overall_unique_setups",
    "overall_dates",
    "overall_hit_rate",
    "overall_avg_pnl",
    "overall_profit_factor",
    "overall_max_drawdown",
    "train_outcomes",
    "train_unique_setups",
    "train_dates",
    "train_hit_rate",
    "train_avg_pnl",
    "train_profit_factor",
    "train_max_drawdown",
    "validation_outcomes",
    "validation_unique_setups",
    "validation_dates",
    "validation_hit_rate",
    "validation_avg_pnl",
    "validation_profit_factor",
    "validation_max_drawdown",
    "worst_pnl",
    "verdict",
]
ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS = [
    "ticker",
    "direction",
    "strategy",
    "horizon_market_days",
    "forward_tests",
    "forward_dates",
    "forward_hit_rate",
    "forward_avg_pnl",
    "forward_profit_factor",
    "forward_worst_pnl",
    "first_forward_date",
    "last_forward_date",
    "verdict",
]
TRACKING_COLUMNS = [
    "trade_id",
    "status",
    "signal_date",
    "last_seen_as_of",
    "ticker",
    "direction",
    "strategy",
    "variant_tag",
    "trade_setup",
    "target_expiry",
    "long_strike",
    "short_strike",
    "spread_width",
    "cost_type",
    "entry_price",
    "max_risk",
    "max_profit",
    "position_size_tier",
    "position_size_guidance",
    "entry_trigger",
    "source_report",
    "outcome_as_of",
    "outcome_exit_date",
    "outcome_final",
    "outcome_status",
    "outcome_verdict",
    "outcome_entry_price",
    "outcome_exit_price",
    "outcome_pnl",
    "outcome_return_on_risk",
    "outcome_days_held",
    "outcome_reason",
    "outcome_source",
]
MIN_CONFIDENCE_TIERS = {
    "PROBE_ONLY": {
        "risk_units": 0.10,
        "guidance": "Probe only: 0.10R / paper or smallest defined-risk test; evidence is still thin.",
    },
    "STARTER_RISK": {
        "risk_units": 0.25,
        "guidance": "Starter only: 0.25R / 1 defined-risk spread max until forward validation improves.",
    },
    "STANDARD_RISK": {
        "risk_units": 0.50,
        "guidance": "Standard: 0.50R max for one defined-risk spread; do not add without a fresh catalyst.",
    },
    "MAX_PLANNED_RISK": {
        "risk_units": 1.00,
        "guidance": "Max planned tier: 1.00R max for one defined-risk trade; never all account capital.",
    },
}

STRATEGY_FAMILY_DESCRIPTIONS = {
    "bull_energy_materials_debit": (
        "Bull call debits in Energy/Basic Materials when price and options flow both confirm."
    ),
    "bear_momentum_put_debit": (
        "Bear put debits when price and options flow both confirm downside momentum."
    ),
    "bull_quality_momentum_debit": (
        "Bull call debits with high score, confirming price, confirming flow, and whale coverage."
    ),
    "bull_oi_dp_confirmed_debit": (
        "Bull call debits where price, flow, OI, and dark-pool accumulation all agree."
    ),
    "bull_credit_put_support": (
        "Bull put credit spreads with bullish flow/OI support."
    ),
    "growth_bull_call_debit": (
        "Bull call debits in high-beta growth sectors; audited separately because this bucket has been loss-prone."
    ),
}


def _looks_like_data_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(p.is_dir() and DATE_RE.match(p.name) for p in path.iterdir())
    except OSError:
        return False


def default_root_dir() -> Path:
    cwd = Path.cwd().resolve()
    if _looks_like_data_root(cwd):
        return cwd
    if _looks_like_data_root(DEFAULT_REPO_ROOT):
        return DEFAULT_REPO_ROOT
    return cwd


def _roll_to_market_day(value: dt.date) -> dt.date:
    if value.weekday() == 5:
        return value - dt.timedelta(days=1)
    if value.weekday() == 6:
        return value - dt.timedelta(days=2)
    return value


def _parse_date(value: str) -> dt.date:
    return _roll_to_market_day(dt.date.fromisoformat(value))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dated-folder trend analysis and emit backtest-gated option trade "
            "recommendations."
        )
    )
    parser.add_argument(
        "tokens",
        nargs="*",
        help=(
            "Convenience form: YYYY-MM-DD [lookback]. Also accepts [lookback] "
            "YYYY-MM-DD."
        ),
    )
    parser.add_argument("--as-of", default="", help="As-of date YYYY-MM-DD.")
    parser.add_argument(
        "--lookback",
        type=_positive_int,
        default=None,
        help=f"Trading-day lookback. Default: {DEFAULT_LOOKBACK}.",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=DEFAULT_TOP,
        help=f"Maximum actionable recommendations. Default: {DEFAULT_TOP}.",
    )
    parser.add_argument(
        "--candidate-top",
        type=_positive_int,
        default=DEFAULT_CANDIDATE_TOP,
        help=f"Maximum high-conviction candidates to show. Default: {DEFAULT_CANDIDATE_TOP}.",
    )
    parser.add_argument(
        "--candidate-min-score",
        type=float,
        default=DEFAULT_CANDIDATE_MIN_SCORE,
        help=f"Minimum swing score for high-conviction candidates. Default: {DEFAULT_CANDIDATE_MIN_SCORE:.1f}.",
    )
    parser.add_argument(
        "--candidate-min-confirmations",
        type=int,
        default=DEFAULT_CANDIDATE_MIN_CONFIRMATIONS,
        help=(
            "Minimum independent trend confirmations for high-conviction "
            f"candidates. Default: {DEFAULT_CANDIDATE_MIN_CONFIRMATIONS}."
        ),
    )
    parser.add_argument(
        "--candidate-max-conflicts",
        type=int,
        default=DEFAULT_CANDIDATE_MAX_CONFLICTS,
        help=(
            "Maximum contradictory signals allowed for high-conviction "
            f"candidates. Default: {DEFAULT_CANDIDATE_MAX_CONFLICTS}."
        ),
    )
    parser.add_argument(
        "--candidate-pool",
        type=_positive_int,
        default=None,
        help=(
            "Number of high-scoring pattern candidates to backtest before the "
            "actionable filter is applied. Default: max(top*3, top+10)."
        ),
    )
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory containing YYYY-MM-DD folders. Default: repo/data root.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <root-dir>/out/trend_analysis.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "rulebook_config_swing_trend.yaml"),
        help="Swing trend YAML config to reuse for scoring.",
    )
    parser.add_argument(
        "--no-schwab",
        action="store_true",
        help="Skip Schwab live chain validation.",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip likelihood backtest. The report will mark trades as patterns only.",
    )
    parser.add_argument(
        "--allow-low-sample",
        action="store_true",
        help="Allow LOW_SAMPLE rows into actionable output when edge is positive.",
    )
    parser.add_argument(
        "--min-backtest-edge",
        type=float,
        default=0.0,
        help="Minimum backtested edge percentage for actionable trades.",
    )
    parser.add_argument(
        "--min-backtest-signals",
        type=int,
        default=100,
        help="Minimum analog signal count for actionable PASS rows.",
    )
    parser.add_argument(
        "--min-workup-signals",
        type=int,
        default=DEFAULT_MIN_WORKUP_SIGNALS,
        help=(
            "Minimum analog signal count for Trade Workup LOW_SAMPLE rows. "
            f"Default: {DEFAULT_MIN_WORKUP_SIGNALS}."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional cache directory used by setup_likelihood_backtest.",
    )
    parser.add_argument(
        "--max-backtest-setups",
        type=_positive_int,
        default=160,
        help=(
            "Maximum concrete trade structures sent to the expensive "
            "likelihood backtest. Default: 160."
        ),
    )
    parser.add_argument(
        "--quote-replay",
        choices=["gate", "diagnostic", "off"],
        default="gate",
        help=(
            "Daily historical option quote replay mode. gate requires replay "
            "PASS/PARTIAL_PASS for actionable trades; diagnostic reports it "
            "without gating; off skips it. Default: gate."
        ),
    )
    parser.add_argument(
        "--quote-replay-web-fallback",
        action="store_true",
        help="Allow web fallback for underlying closes when replay settles by expiry intrinsic.",
    )
    parser.add_argument(
        "--allow-earnings-risk",
        action="store_true",
        help="Allow trades whose earnings window overlaps the option holding period.",
    )
    parser.add_argument(
        "--allow-volatile-ic",
        action="store_true",
        help="Allow iron condors when live GEX regime is volatile.",
    )
    parser.add_argument(
        "--max-bid-ask-to-price-pct",
        type=float,
        default=DEFAULT_MAX_BID_ASK_TO_PRICE_PCT,
        help=(
            "Maximum average leg bid/ask width divided by spread credit/debit "
            f"for actionable trades. Default: {DEFAULT_MAX_BID_ASK_TO_PRICE_PCT:.2f}."
        ),
    )
    parser.add_argument(
        "--max-bid-ask-to-width-pct",
        type=float,
        default=DEFAULT_MAX_BID_ASK_TO_WIDTH_PCT,
        help=(
            "Maximum average leg bid/ask width divided by spread width for "
            f"actionable trades. Default: {DEFAULT_MAX_BID_ASK_TO_WIDTH_PCT:.2f}."
        ),
    )
    parser.add_argument(
        "--max-short-delta",
        type=float,
        default=DEFAULT_MAX_SHORT_DELTA,
        help=f"Maximum absolute short-leg delta for actionable credit trades. Default: {DEFAULT_MAX_SHORT_DELTA:.2f}.",
    )
    parser.add_argument(
        "--min-underlying-price",
        type=float,
        default=DEFAULT_MIN_UNDERLYING_PRICE,
        help=f"Minimum underlying price for actionable trend trades. Default: {DEFAULT_MIN_UNDERLYING_PRICE:.2f}.",
    )
    parser.add_argument(
        "--min-debit-spread-price",
        type=float,
        default=DEFAULT_MIN_DEBIT_SPREAD_PRICE,
        help=f"Minimum live debit spread price for actionable debit trades. Default: {DEFAULT_MIN_DEBIT_SPREAD_PRICE:.2f}.",
    )
    parser.add_argument(
        "--min-whale-appearances",
        type=int,
        default=DEFAULT_MIN_WHALE_APPEARANCES,
        help=(
            "Maximum required whale-mention days for actionable directional "
            "trend trades. The effective threshold scales down for short "
            f"lookbacks. Default cap: {DEFAULT_MIN_WHALE_APPEARANCES}."
        ),
    )
    parser.add_argument(
        "--allow-flow-conflict",
        action="store_true",
        help="Allow directional trades when flow direction conflicts with the trade direction.",
    )
    parser.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Skip market-regime compatibility filtering.",
    )
    parser.add_argument(
        "--position-json",
        default="",
        help=(
            "Optional trade-desk position_data JSON for open-position awareness. "
            "Default: latest out/trade_analysis/position_data_*.json when present."
        ),
    )
    parser.add_argument(
        "--no-position-check",
        action="store_true",
        help="Skip open-position duplicate/conflict checks.",
    )
    parser.add_argument(
        "--trade-tracker",
        default="",
        help=(
            "CSV used to track newly emitted actionable trades. "
            f"Default: <out-dir>/{DEFAULT_TRACKING_FILE_NAME}."
        ),
    )
    parser.add_argument(
        "--no-trade-tracking",
        action="store_true",
        help="Do not append actionable trades to the post-trade tracker CSV.",
    )
    parser.add_argument(
        "--no-outcome-update",
        action="store_true",
        help="Do not refresh post-trade tracker outcomes from local option snapshots.",
    )
    parser.add_argument(
        "--walk-forward-samples",
        type=int,
        default=DEFAULT_WALK_FORWARD_SAMPLES,
        help=(
            "Number of prior signal dates to replay for confidence audit. "
            "Use 0 to skip. Default: "
            f"{DEFAULT_WALK_FORWARD_SAMPLES}."
        ),
    )
    parser.add_argument(
        "--walk-forward-horizons",
        default=DEFAULT_WALK_FORWARD_HORIZONS,
        help=(
            "Comma-separated market-day holding horizons for the walk-forward "
            f"audit. Default: {DEFAULT_WALK_FORWARD_HORIZONS}."
        ),
    )
    parser.add_argument(
        "--walk-forward-top",
        type=_positive_int,
        default=DEFAULT_WALK_FORWARD_TOP,
        help=(
            "Top actionable historical candidates per signal date to score in "
            f"the walk-forward audit. Default: {DEFAULT_WALK_FORWARD_TOP}."
        ),
    )
    parser.add_argument(
        "--reuse-walk-forward-raw",
        action="store_true",
        help=(
            "Reuse existing per-date walk-forward raw CSVs when present. "
            "Use this for rerunning audit aggregation without rebuilding every historical candidate file."
        ),
    )
    parser.add_argument(
        "--reuse-walk-forward-outcomes",
        action="store_true",
        help=(
            "Reuse an existing walk-forward outcomes CSV when present. "
            "Use only for exact reruns with the same output directory and audit settings."
        ),
    )
    parser.add_argument(
        "--reuse-research-outcomes",
        action="store_true",
        help=(
            "Reuse an existing detailed research outcomes CSV when present. "
            "Use only for exact reruns with the same output directory and audit settings."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_invocation(args: argparse.Namespace) -> Tuple[Optional[dt.date], int]:
    as_of: Optional[dt.date] = _parse_date(args.as_of) if str(args.as_of or "").strip() else None
    lookback: Optional[int] = args.lookback

    for token in args.tokens:
        value = str(token).strip()
        if DATE_RE.match(value):
            if as_of is not None and as_of != _parse_date(value):
                raise ValueError(f"Multiple as-of dates supplied: {as_of} and {value}")
            as_of = _parse_date(value)
        elif re.fullmatch(r"\d+", value):
            parsed = int(value)
            if parsed <= 0:
                raise ValueError(f"Lookback must be positive: {value}")
            if lookback is not None and lookback != parsed:
                raise ValueError(f"Multiple lookbacks supplied: {lookback} and {parsed}")
            lookback = parsed
        else:
            raise ValueError(f"Unrecognized trend-analysis argument: {value}")

    return as_of, int(lookback or DEFAULT_LOOKBACK)


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return math.nan
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: Any) -> int:
    parsed = _safe_float(value)
    if not math.isfinite(parsed):
        return 0
    return int(round(parsed))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ok"}


def _truthy_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].apply(_truthy)


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"false", "0", "no", "n", "fail"}


def _safe_pct_return(close: Any, prev_close: Any) -> float:
    c = _safe_float(close)
    p = _safe_float(prev_close)
    if not math.isfinite(c) or not math.isfinite(p) or p == 0:
        return math.nan
    return (c - p) / p


def _parse_tracker_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _latest_position_json(root: Path) -> Optional[Path]:
    trade_dir = root / "out" / "trade_analysis"
    if not trade_dir.exists():
        return None
    paths = sorted(trade_dir.glob("position_data_*.json"))
    return paths[-1] if paths else None


def _position_direction(pos: Dict[str, Any]) -> str:
    qty = _safe_float(pos.get("qty"))
    right = str(pos.get("put_call", "") or "").strip().upper()
    if not math.isfinite(qty) or right not in {"CALL", "PUT"}:
        return "unknown"
    if (qty > 0 and right == "CALL") or (qty < 0 and right == "PUT"):
        return "bullish"
    if (qty > 0 and right == "PUT") or (qty < 0 and right == "CALL"):
        return "bearish"
    return "unknown"


def _load_open_position_exposure(position_json: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    if not position_json or not position_json.exists():
        return {}
    try:
        payload = json.loads(position_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    exposure: Dict[str, List[Dict[str, Any]]] = {}
    for pos in payload.get("positions", []) or []:
        if str(pos.get("asset_type", "") or "").upper() != "OPTION":
            continue
        underlying = str(pos.get("underlying", "") or "").strip().upper()
        if not underlying:
            symbol = str(pos.get("symbol", "") or "").strip().upper()
            underlying = symbol.split()[0] if symbol else ""
        if not underlying:
            continue
        exposure.setdefault(underlying, []).append(
            {
                "symbol": str(pos.get("symbol", "") or "").strip(),
                "direction": _position_direction(pos),
                "expiry": str(pos.get("expiry", "") or "").strip(),
                "qty": pos.get("qty", ""),
            }
        )
    return exposure


def annotate_open_position_awareness(
    candidates: pd.DataFrame,
    position_json: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if candidates.empty:
        return candidates.copy(), {
            "checked": bool(position_json),
            "position_json": str(position_json or ""),
            "open_underlyings": 0,
            "blocked_rows": 0,
        }
    out = candidates.copy()
    exposure = _load_open_position_exposure(position_json)
    out["open_position_gate_pass"] = True
    out["open_position_status"] = "clear" if exposure else ("not_checked" if not position_json else "no_open_options")
    out["open_position_summary"] = ""
    blocked = 0
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        positions = exposure.get(ticker, [])
        if not positions:
            continue
        blocked += 1
        directions = sorted({p.get("direction", "unknown") for p in positions})
        symbols = [p.get("symbol", "") for p in positions[:4] if p.get("symbol")]
        out.at[idx, "open_position_gate_pass"] = False
        out.at[idx, "open_position_status"] = "open_option_exposure"
        out.at[idx, "open_position_summary"] = (
            f"existing option exposure in {ticker}: {len(positions)} leg(s), "
            f"directions {','.join(directions) or 'unknown'}"
            + (f"; {', '.join(symbols)}" if symbols else "")
        )
    return out, {
        "checked": bool(position_json),
        "position_json": str(position_json or ""),
        "open_underlyings": len(exposure),
        "blocked_rows": blocked,
    }


def compute_market_regime(root: Path, trading_days: List[Tuple[dt.date, Path]]) -> Dict[str, Any]:
    if not trading_days:
        return {"regime": "unknown", "reason": "no trading days"}
    day, day_dir = trading_days[-1]
    screener_path = swing.resolve_csv_for_day(day_dir, day.isoformat(), "stock-screener")
    if not screener_path:
        return {"regime": "unknown", "reason": "latest stock-screener file missing"}
    try:
        df = pd.read_csv(screener_path, low_memory=False)
    except Exception as exc:
        return {"regime": "unknown", "reason": f"stock-screener read failed: {exc}"}
    if df.empty:
        return {"regime": "unknown", "reason": "latest stock-screener file empty"}
    tickers = df.get("ticker", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    common = df.copy()
    if "issue_type" in common.columns:
        issue = common["issue_type"].fillna("").astype(str).str.lower()
        common = common[issue.eq("common stock")].copy()
    returns = common.apply(lambda r: _safe_pct_return(r.get("close"), r.get("prev_close")), axis=1)
    returns = returns[np.isfinite(returns)]
    breadth = float(returns.gt(0).mean()) if len(returns) else math.nan

    def index_row(symbol: str) -> Optional[pd.Series]:
        subset = df[tickers.eq(symbol)]
        return subset.iloc[0] if not subset.empty else None

    spy = index_row("SPY")
    qqq = index_row("QQQ")
    spy_ret = _safe_pct_return(spy.get("close"), spy.get("prev_close")) if spy is not None else math.nan
    qqq_ret = _safe_pct_return(qqq.get("close"), qqq.get("prev_close")) if qqq is not None else math.nan
    index_pcr_values = []
    index_iv_values = []
    for row in (spy, qqq):
        if row is None:
            continue
        pcr = _safe_float(row.get("put_call_ratio"))
        iv = _safe_float(row.get("iv_rank"))
        if math.isfinite(pcr):
            index_pcr_values.append(pcr)
        if math.isfinite(iv):
            index_iv_values.append(iv)
    index_pcr = float(np.mean(index_pcr_values)) if index_pcr_values else math.nan
    index_iv_rank = float(np.mean(index_iv_values)) if index_iv_values else math.nan

    if (
        (math.isfinite(breadth) and breadth < 0.40)
        or (math.isfinite(spy_ret) and spy_ret < -0.005 and math.isfinite(qqq_ret) and qqq_ret < -0.005)
        or (math.isfinite(index_pcr) and index_pcr >= 1.25)
    ):
        regime = "risk_off"
    elif (
        math.isfinite(breadth)
        and breadth >= 0.52
        and math.isfinite(spy_ret)
        and spy_ret > 0.003
        and math.isfinite(qqq_ret)
        and qqq_ret > 0.003
        and (not math.isfinite(index_pcr) or index_pcr < 1.20)
    ):
        regime = "risk_on"
    else:
        regime = "mixed"
    return {
        "regime": regime,
        "as_of": day.isoformat(),
        "breadth": breadth,
        "spy_return_pct": spy_ret * 100 if math.isfinite(spy_ret) else math.nan,
        "qqq_return_pct": qqq_ret * 100 if math.isfinite(qqq_ret) else math.nan,
        "index_put_call_ratio": index_pcr,
        "index_iv_rank": index_iv_rank,
        "reason": (
            f"breadth {_fmt_num(breadth * 100 if math.isfinite(breadth) else math.nan, 0)}%, "
            f"SPY {_fmt_num(spy_ret * 100 if math.isfinite(spy_ret) else math.nan, 2)}%, "
            f"QQQ {_fmt_num(qqq_ret * 100 if math.isfinite(qqq_ret) else math.nan, 2)}%, "
            f"index PCR {_fmt_num(index_pcr, 2)}"
        ),
    }


def annotate_regime_filter(candidates: pd.DataFrame, regime: Dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    label = str(regime.get("regime", "unknown") or "unknown")
    out["market_regime"] = label
    out["market_regime_summary"] = str(regime.get("reason", "") or "")
    out["market_regime_gate_pass"] = True
    for idx, row in out.iterrows():
        direction = str(row.get("direction", "") or "").strip().lower()
        strategy = str(row.get("strategy", "") or "").strip().lower()
        block = False
        if label == "risk_off" and direction == "bullish" and "debit" in strategy:
            block = True
        elif label == "risk_on" and direction == "bearish" and "debit" in strategy:
            block = True
        elif label == "risk_off" and "iron condor" in strategy:
            block = True
        if block:
            out.at[idx, "market_regime_gate_pass"] = False
            out.at[idx, "market_regime_summary"] = (
                f"{label} conflict for {direction} {row.get('strategy', '')}: "
                f"{regime.get('reason', '')}"
            )
    return out


def _effective_min_whale_appearances(row: pd.Series, configured_min: int) -> int:
    configured = max(0, int(configured_min))
    if configured <= 0:
        return 0
    observed = _safe_int(row.get("days_observed"))
    if observed <= 0:
        return configured
    # An absolute 8-day threshold makes 5-day trend windows impossible. Keep
    # the configured threshold as the normal cap, but scale down for short
    # windows so a dense 4/5 whale signal can pass while 5/30 still fails.
    scaled = max(3, int(math.ceil(observed * 0.60)))
    return min(configured, scaled)


def _first_finite(row: pd.Series, columns: Sequence[str]) -> float:
    for col in columns:
        if col in row.index:
            value = _safe_float(row.get(col))
            if math.isfinite(value):
                return value
    return math.nan


def _is_directional_debit(strategy: str, direction: str) -> bool:
    return "debit" in strategy and direction in {"bullish", "bearish"}


def backtest_passes(
    row: pd.Series,
    *,
    min_edge: float,
    min_signals: int,
    allow_low_sample: bool,
) -> bool:
    verdict = str(row.get("backtest_verdict", "") or "").strip().upper()
    edge = _safe_float(row.get("edge_pct"))
    signals = _safe_int(row.get("backtest_signals"))

    if verdict == "PASS":
        if math.isfinite(edge) and edge < min_edge:
            return False
        return signals >= max(0, int(min_signals))

    if allow_low_sample and verdict == "LOW_SAMPLE":
        return math.isfinite(edge) and edge >= min_edge and signals > 0

    return False


def _ticker_playbook_support_passes(row: pd.Series, *, min_edge: float) -> bool:
    if not _truthy(row.get("ticker_playbook_gate_pass")):
        return False
    if "rolling_playbook_gate_pass" in row.index and not _truthy(row.get("rolling_playbook_gate_pass")):
        return False
    verdict = str(row.get("backtest_verdict", "") or "").strip().upper()
    edge = _safe_float(row.get("edge_pct"))
    signals = _safe_int(row.get("backtest_signals"))
    if math.isfinite(edge) and edge < float(min_edge):
        return False
    if verdict == "PASS":
        return signals > 0
    if verdict == "LOW_SAMPLE":
        return signals >= DEFAULT_MIN_WORKUP_SIGNALS
    return False


def historical_support_passes(
    row: pd.Series,
    *,
    min_edge: float,
    min_signals: int,
    allow_low_sample: bool,
) -> bool:
    if backtest_passes(
        row,
        min_edge=min_edge,
        min_signals=min_signals,
        allow_low_sample=allow_low_sample,
    ):
        return True
    return _ticker_playbook_support_passes(row, min_edge=min_edge)


def research_gate_passes(row: pd.Series) -> bool:
    family_present = "strategy_family_gate_pass" in row.index
    playbook_present = "ticker_playbook_gate_pass" in row.index
    if not family_present and not playbook_present:
        return True
    if "rolling_playbook_gate_pass" in row.index and not _truthy(row.get("rolling_playbook_gate_pass")):
        return False
    return _truthy(row.get("strategy_family_gate_pass")) or _truthy(row.get("ticker_playbook_gate_pass"))


def base_gate_reasons(
    row: pd.Series,
    *,
    backtest_enabled: bool,
    schwab_enabled: bool,
    quote_replay_mode: str,
    min_edge: float,
    min_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
) -> List[str]:
    reasons: List[str] = []
    score = _safe_float(row.get("swing_score"))
    if float(min_swing_score) > 0:
        if not math.isfinite(score) or score < float(min_swing_score):
            reasons.append(f"swing score {_fmt_num(score, 1)} < min {float(min_swing_score):.1f}")

    if not backtest_enabled:
        reasons.append("backtest skipped")
    elif not historical_support_passes(
        row,
        min_edge=min_edge,
        min_signals=min_signals,
        allow_low_sample=allow_low_sample,
    ):
        verdict = _verdict_text(row.get("backtest_verdict", ""))
        edge = _safe_float(row.get("edge_pct"))
        signals = _safe_int(row.get("backtest_signals"))
        detail = f"backtest {verdict}"
        if math.isfinite(edge):
            detail += f", edge {edge:.1f}%"
        detail += f", signals {signals}"
        if verdict == "PASS" and signals < int(min_signals):
            detail += f" < min {int(min_signals)}"
        reasons.append(detail)
    if schwab_enabled and not _truthy(row.get("live_validated")):
        note = str(row.get("live_validation_note", "") or "").strip()
        reasons.append(f"Schwab live failed{': ' + note if note else ''}")

    if str(quote_replay_mode or "off").lower() == "gate":
        if not trend_quote_replay.quote_replay_passes(row):
            verdict = str(row.get("quote_replay_verdict", "") or "UNAVAILABLE").strip().upper()
            status = str(row.get("quote_replay_status", "") or "").strip()
            reason = str(row.get("quote_replay_reason", "") or "").strip()
            detail = f"quote replay {verdict}"
            if status:
                detail += f", status {status}"
            if reason:
                detail += f": {reason}"
            reasons.append(detail)

    if "market_regime_gate_pass" in row.index and not _truthy(row.get("market_regime_gate_pass")):
        summary = str(row.get("market_regime_summary", "") or "").strip()
        reasons.append(f"market regime conflict{': ' + summary if summary else ''}")

    if "open_position_gate_pass" in row.index and not _truthy(row.get("open_position_gate_pass")):
        summary = str(row.get("open_position_summary", "") or "").strip()
        reasons.append(f"open position conflict{': ' + summary if summary else ''}")

    if "rolling_playbook_gate_pass" in row.index and not _truthy(row.get("rolling_playbook_gate_pass")):
        summary = str(row.get("rolling_playbook_summary", "") or "").strip()
        reasons.append(f"rolling playbook forward validation failed{': ' + summary if summary else ''}")

    if (
        ("strategy_family_gate_pass" in row.index or "ticker_playbook_gate_pass" in row.index)
        and not research_gate_passes(row)
    ):
        family = str(row.get("strategy_family", "") or "no matching family").strip()
        verdict = str(row.get("strategy_family_verdict", "") or "not_promotable").strip()
        horizon = _safe_int(row.get("strategy_family_horizon"))
        detail = f"strategy family audit not promotable: {family} ({verdict}"
        if horizon > 0:
            detail += f", {horizon}d"
        detail += ")"
        summary = str(row.get("strategy_family_summary", "") or "").strip()
        if summary:
            detail += f": {summary}"
        playbook_summary = str(row.get("ticker_playbook_summary", "") or "").strip()
        if playbook_summary:
            detail += f"; ticker playbook: {playbook_summary}"
        reasons.append(detail)

    return reasons


def quality_gate_reasons(
    row: pd.Series,
    *,
    schwab_enabled: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
) -> List[str]:
    reasons: List[str] = []
    strategy = str(row.get("strategy", "") or "").strip().lower()
    direction = str(row.get("direction", "") or "").strip().lower()

    latest_close = _safe_float(row.get("latest_close"))
    if float(min_underlying_price) > 0:
        if math.isfinite(latest_close) and latest_close < float(min_underlying_price):
            reasons.append(
                f"lotto underlying: stock ${latest_close:.2f} < min ${float(min_underlying_price):.2f}"
            )
        elif not math.isfinite(latest_close):
            reasons.append("missing underlying price for professional-quality gate")

    if int(min_whale_appearances) > 0 and direction in {"bullish", "bearish"}:
        effective_min_whales = _effective_min_whale_appearances(row, int(min_whale_appearances))
        whales = _safe_int(row.get("whale_appearances"))
        if whales < effective_min_whales:
            reasons.append(
                f"thin institutional confirmation: whale days {whales} < min {effective_min_whales}"
            )

    if not allow_flow_conflict and direction in {"bullish", "bearish"}:
        flow = str(row.get("flow_direction", "") or "").strip().lower()
        if flow in {"bullish", "bearish"} and flow != direction:
            reasons.append(f"flow conflict: {flow} flow vs {direction} trade")

    if _is_directional_debit(strategy, direction):
        price_direction = str(row.get("price_direction", "") or "").strip().lower()
        price_score = _safe_float(row.get("price_trend"))
        if price_direction != direction:
            label = price_direction or "missing"
            reasons.append(f"directional price not confirming: {label} vs {direction} trade")
        if not math.isfinite(price_score) or price_score < DEFAULT_MIN_DIRECTIONAL_PRICE_SCORE:
            score_text = _fmt_num(price_score, 1)
            reasons.append(
                f"weak directional price trend: {score_text} < min {DEFAULT_MIN_DIRECTIONAL_PRICE_SCORE:.1f}"
            )

        if not allow_flow_conflict:
            flow_direction = str(row.get("flow_direction", "") or "").strip().lower()
            flow_score = _safe_float(row.get("flow_persistence"))
            if flow_direction != direction:
                label = flow_direction or "missing"
                reasons.append(f"directional flow not confirming: {label} vs {direction} trade")
            if not math.isfinite(flow_score) or flow_score < DEFAULT_MIN_DIRECTIONAL_FLOW_SCORE:
                score_text = _fmt_num(flow_score, 1)
                reasons.append(
                    f"weak directional flow: {score_text} < min {DEFAULT_MIN_DIRECTIONAL_FLOW_SCORE:.1f}"
                )

        spread_width = abs(_safe_float(row.get("spread_width")))
        debit = abs(
            _first_finite(
                row,
                ("live_spread_cost", "quote_replay_entry_net", "est_cost"),
            )
        )
        if math.isfinite(spread_width) and spread_width > 0 and math.isfinite(debit) and debit > 0:
            debit_to_width = debit / spread_width
            if debit_to_width > DEFAULT_MAX_DEBIT_TO_WIDTH_PCT:
                reasons.append(
                    f"expensive debit: entry {debit_to_width:.0%} of spread width > {DEFAULT_MAX_DEBIT_TO_WIDTH_PCT:.0%}"
                )

        long_strike = _first_finite(row, ("live_long_strike", "long_strike"))
        if math.isfinite(latest_close) and latest_close > 0 and math.isfinite(long_strike):
            if direction == "bullish":
                otm_pct = (long_strike - latest_close) / latest_close
            else:
                otm_pct = (latest_close - long_strike) / latest_close
            if otm_pct > DEFAULT_MAX_LONG_STRIKE_OTM_PCT:
                reasons.append(
                    f"long strike too far OTM: {otm_pct:.0%} > {DEFAULT_MAX_LONG_STRIKE_OTM_PCT:.0%}"
                )

    if not allow_earnings_risk and _falsey(row.get("earnings_safe")):
        label = str(row.get("earnings_label", "") or "").strip()
        if "after expiry" in label:
            reasons.append(f"earnings buffer risk{': ' + label if label else ''}")
        else:
            reasons.append(f"earnings in trade window{': ' + label if label else ''}")

    if schwab_enabled:
        ba_width = _safe_float(row.get("live_bid_ask_width"))
        live_price = abs(_safe_float(row.get("live_spread_cost")))
        spread_width = abs(_safe_float(row.get("spread_width")))
        if math.isfinite(ba_width) and ba_width > 0:
            if math.isfinite(live_price) and live_price > 0:
                ba_to_price = ba_width / live_price
                if ba_to_price > max_bid_ask_to_price_pct:
                    reasons.append(
                        f"wide market: bid/ask {ba_to_price:.0%} of spread price"
                    )
            else:
                reasons.append("missing live spread price for liquidity gate")
            if math.isfinite(spread_width) and spread_width > 0:
                ba_to_width = ba_width / spread_width
                if ba_to_width > max_bid_ask_to_width_pct:
                    reasons.append(
                        f"wide market: bid/ask {ba_to_width:.0%} of spread width"
                    )
        else:
            reasons.append("missing bid/ask width")

    if "debit" in strategy and float(min_debit_spread_price) > 0:
        live_price = _safe_float(row.get("live_spread_cost"))
        if math.isfinite(live_price) and live_price < float(min_debit_spread_price):
            reasons.append(
                f"lotto debit: spread price ${live_price:.2f} < min ${float(min_debit_spread_price):.2f}"
            )
        elif not math.isfinite(live_price):
            est_price = _safe_float(row.get("est_cost"))
            if math.isfinite(est_price) and est_price < float(min_debit_spread_price):
                reasons.append(
                    f"lotto debit: estimated spread price ${est_price:.2f} < min ${float(min_debit_spread_price):.2f}"
                )

    if max_short_delta > 0:
        if "iron condor" in strategy:
            for label, col in (
                ("short put delta", "short_put_delta_live"),
                ("short call delta", "short_call_delta_live"),
            ):
                delta = _safe_float(row.get(col))
                if math.isfinite(delta) and abs(delta) > max_short_delta:
                    reasons.append(f"{label} {delta:.2f} exceeds {max_short_delta:.2f}")
        elif "credit" in strategy:
            delta = _safe_float(row.get("short_delta_live"))
            if math.isfinite(delta) and abs(delta) > max_short_delta:
                reasons.append(f"short delta {delta:.2f} exceeds {max_short_delta:.2f}")

    if not allow_volatile_ic and "iron condor" in strategy:
        gex_regime = str(row.get("gex_regime", "") or "").strip().lower()
        if gex_regime == "volatile":
            reasons.append("volatile GEX regime for iron condor")

    return reasons


def split_actionable_candidates(
    candidates: pd.DataFrame,
    *,
    top_n: int,
    backtest_enabled: bool,
    min_edge: float,
    min_signals: int,
    allow_low_sample: bool,
    schwab_enabled: bool = False,
    quote_replay_mode: str = "off",
    min_swing_score: float = 0.0,
    allow_earnings_risk: bool = False,
    allow_volatile_ic: bool = False,
    allow_flow_conflict: bool = False,
    max_bid_ask_to_price_pct: float = DEFAULT_MAX_BID_ASK_TO_PRICE_PCT,
    max_bid_ask_to_width_pct: float = DEFAULT_MAX_BID_ASK_TO_WIDTH_PCT,
    max_short_delta: float = DEFAULT_MAX_SHORT_DELTA,
    min_underlying_price: float = 0.0,
    min_debit_spread_price: float = 0.0,
    min_whale_appearances: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy()

    df = candidates.copy()
    if "swing_score" in df.columns:
        df["_sort_score"] = pd.to_numeric(df["swing_score"], errors="coerce").fillna(-1)
    else:
        df["_sort_score"] = 0.0
    if "edge_pct" in df.columns:
        df["_sort_edge"] = pd.to_numeric(df["edge_pct"], errors="coerce").fillna(-999)
    else:
        df["_sort_edge"] = -999.0
    df["_trend_row_id"] = range(len(df))

    df["base_gate_reasons"] = df.apply(
        lambda r: "; ".join(
            base_gate_reasons(
                r,
                backtest_enabled=backtest_enabled,
                schwab_enabled=schwab_enabled,
                quote_replay_mode=quote_replay_mode,
                min_edge=min_edge,
                min_signals=min_signals,
                min_swing_score=min_swing_score,
                allow_low_sample=allow_low_sample,
            )
        ),
        axis=1,
    )
    df["quality_reject_reasons"] = df.apply(
        lambda r: "; ".join(
            quality_gate_reasons(
                r,
                schwab_enabled=schwab_enabled,
                allow_earnings_risk=allow_earnings_risk,
                allow_volatile_ic=allow_volatile_ic,
                allow_flow_conflict=allow_flow_conflict,
                max_bid_ask_to_price_pct=max_bid_ask_to_price_pct,
                max_bid_ask_to_width_pct=max_bid_ask_to_width_pct,
                max_short_delta=max_short_delta,
                min_underlying_price=min_underlying_price,
                min_debit_spread_price=min_debit_spread_price,
                min_whale_appearances=min_whale_appearances,
            )
        ),
        axis=1,
    )
    df["backtest_gate_pass"] = df.apply(
        lambda r: backtest_enabled
        and historical_support_passes(
            r,
            min_edge=min_edge,
            min_signals=min_signals,
            allow_low_sample=allow_low_sample,
        ),
        axis=1,
    )
    if not schwab_enabled:
        df["schwab_gate_pass"] = True
    elif "live_validated" in df.columns:
        df["schwab_gate_pass"] = df["live_validated"].apply(_truthy)
    else:
        df["schwab_gate_pass"] = False
    df["quote_replay_gate_pass"] = (
        True
        if str(quote_replay_mode or "off").lower() != "gate"
        else df.apply(trend_quote_replay.quote_replay_passes, axis=1)
    )
    df["base_gate_pass"] = df["base_gate_reasons"].astype(str).str.strip().eq("")
    df["quality_gate_pass"] = df["quality_reject_reasons"].astype(str).str.strip().eq("")
    df["actionability_reject_reasons"] = (
        df["base_gate_reasons"].fillna("").astype(str)
        + np.where(
            df["base_gate_reasons"].fillna("").astype(str).str.strip().ne("")
            & df["quality_reject_reasons"].fillna("").astype(str).str.strip().ne(""),
            "; ",
            "",
        )
        + df["quality_reject_reasons"].fillna("").astype(str)
    ).str.strip("; ")
    mask = df["base_gate_pass"] & df["quality_gate_pass"]

    actionable = df[mask].sort_values(
        ["_sort_score", "_sort_edge"], ascending=[False, False]
    )
    actionable = _dedupe_trade_rows(actionable)
    actionable = actionable.head(max(1, int(top_n)))
    patterns = df[~mask].sort_values(
        ["_sort_score", "_sort_edge"], ascending=[False, False]
    )

    return (
        actionable.drop(columns=["_sort_score", "_sort_edge"], errors="ignore").reset_index(drop=True),
        patterns.drop(columns=["_sort_score", "_sort_edge"], errors="ignore").reset_index(drop=True),
    )


def _supports_direction(signal: str, direction: str, *, dp: bool = False) -> bool:
    sig = str(signal or "").strip().lower()
    direc = str(direction or "").strip().lower()
    if dp:
        return (direc == "bullish" and sig == "accumulation") or (
            direc == "bearish" and sig == "distribution"
        )
    return sig == direc


def _conflicts_direction(signal: str, direction: str, *, dp: bool = False) -> bool:
    sig = str(signal or "").strip().lower()
    direc = str(direction or "").strip().lower()
    if dp:
        return (direc == "bullish" and sig == "distribution") or (
            direc == "bearish" and sig == "accumulation"
        )
    return (direc == "bullish" and sig == "bearish") or (
        direc == "bearish" and sig == "bullish"
    )


def candidate_evidence(row: pd.Series) -> Tuple[List[str], List[str]]:
    direction = str(row.get("direction", "") or "").strip().lower()
    confirmations: List[str] = []
    conflicts: List[str] = []
    if direction not in {"bullish", "bearish"}:
        return confirmations, ["no directional bias"]

    score = _safe_float(row.get("swing_score"))
    if math.isfinite(score) and score >= DEFAULT_CANDIDATE_MIN_SCORE:
        confirmations.append(f"score {score:.1f}")

    price_score = _safe_float(row.get("price_trend"))
    price_direction = str(row.get("price_direction", "") or "").strip().lower()
    if _supports_direction(price_direction, direction) and price_score >= 55:
        confirmations.append(f"price trend {price_direction} ({price_score:.0f})")
    elif _conflicts_direction(row.get("price_direction"), direction):
        score_note = f" ({price_score:.0f})" if math.isfinite(price_score) else ""
        conflicts.append(f"price divergence: {price_direction}{score_note} vs {direction} setup")

    flow_score = _safe_float(row.get("flow_persistence"))
    if _supports_direction(row.get("flow_direction"), direction) and flow_score >= 60:
        confirmations.append(f"flow {str(row.get('flow_direction')).lower()} ({flow_score:.0f})")
    elif _conflicts_direction(row.get("flow_direction"), direction):
        conflicts.append(f"flow {str(row.get('flow_direction')).lower()}")

    oi_score = _safe_float(row.get("oi_momentum"))
    if _supports_direction(row.get("oi_direction"), direction) and oi_score >= 55:
        confirmations.append(f"OI {str(row.get('oi_direction')).lower()} ({oi_score:.0f})")
    elif _conflicts_direction(row.get("oi_direction"), direction):
        conflicts.append(f"OI {str(row.get('oi_direction')).lower()}")

    dp_score = _safe_float(row.get("dp_confirmation"))
    if _supports_direction(row.get("dp_direction"), direction, dp=True) and dp_score >= 60:
        confirmations.append(f"DP {str(row.get('dp_direction')).lower()} ({dp_score:.0f})")
    elif _conflicts_direction(row.get("dp_direction"), direction, dp=True):
        conflicts.append(f"DP {str(row.get('dp_direction')).lower()}")

    whale_days = _safe_int(row.get("whale_appearances"))
    observed = max(1, _safe_int(row.get("days_observed")))
    whale_threshold = max(5, int(math.ceil(observed * 0.20)))
    if whale_days >= whale_threshold:
        confirmations.append(f"whale presence {whale_days}d")

    return confirmations, conflicts


def _sort_trade_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_edge_sort"] = (
        pd.to_numeric(out["edge_pct"], errors="coerce").fillna(-999)
        if "edge_pct" in out.columns
        else pd.Series(-999, index=out.index)
    )
    out["_score_sort"] = (
        pd.to_numeric(out["swing_score"], errors="coerce").fillna(-1)
        if "swing_score" in out.columns
        else pd.Series(-1, index=out.index)
    )
    out["_signals_sort"] = (
        pd.to_numeric(out["backtest_signals"], errors="coerce").fillna(0)
        if "backtest_signals" in out.columns
        else pd.Series(0, index=out.index)
    )
    return out.sort_values(
        ["_edge_sort", "_score_sort", "_signals_sort"],
        ascending=[False, False, False],
    ).drop(columns=["_edge_sort", "_score_sort", "_signals_sort"], errors="ignore")


def _dedupe_trade_rows(df: pd.DataFrame, *, by_ticker: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = (
        ("ticker",)
        if by_ticker
        else (
            "ticker",
            "strategy",
            "target_expiry",
            "live_strike_setup",
            "strike_setup",
        )
    )
    dedupe_cols = [c for c in preferred if c in df.columns]
    if not dedupe_cols:
        return df
    return df.drop_duplicates(subset=dedupe_cols, keep="first")


HARD_QUALITY_REJECTS = (
    "lotto underlying",
    "lotto debit",
    "flow conflict",
    "thin institutional confirmation",
)


def _has_hard_quality_reject(row: pd.Series) -> bool:
    reasons = (
        str(row.get("quality_reject_reasons", "") or "")
        + "; "
        + str(row.get("actionability_reject_reasons", "") or "")
    ).lower()
    return any(token in reasons for token in HARD_QUALITY_REJECTS)


def _best_candidate_trade_row(ticker_rows: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    if ticker_rows.empty:
        return None, "none"

    backtest_pass = _truthy_mask(ticker_rows, "backtest_gate_pass")
    schwab_pass = _truthy_mask(ticker_rows, "schwab_gate_pass")
    quote_pass = _truthy_mask(ticker_rows, "quote_replay_gate_pass")
    quality_pass = _truthy_mask(ticker_rows, "quality_gate_pass")
    family_pass = ticker_rows.apply(research_gate_passes, axis=1)
    hard_reject = ticker_rows.apply(_has_hard_quality_reject, axis=1)

    tiers = [
        ("actionable", backtest_pass & schwab_pass & quote_pass & quality_pass & family_pass),
        ("risk_blocked", backtest_pass & schwab_pass & quote_pass & ~quality_pass & ~hard_reject & family_pass),
        ("quote_blocked", backtest_pass & schwab_pass & ~quote_pass & ~hard_reject & family_pass),
        ("live_blocked", backtest_pass & ~schwab_pass & ~hard_reject & family_pass),
    ]
    for label, mask in tiers:
        rows = _sort_trade_rows(ticker_rows[mask])
        if not rows.empty:
            return rows.iloc[0], label
    return None, "none"


def _workup_reason(row: pd.Series, *, min_signals: int) -> str:
    verdict = _verdict_text(row.get("backtest_verdict", ""))
    edge = _fmt_num(row.get("edge_pct"), 1)
    signals = _safe_int(row.get("backtest_signals"))
    if verdict == "LOW_SAMPLE":
        return f"needs more analog history: LOW_SAMPLE, edge {edge}%, signals {signals} < actionable min {int(min_signals)}"
    if verdict == "PASS" and signals < int(min_signals):
        return f"PASS but sample shortfall: signals {signals} < actionable min {int(min_signals)}"
    return f"not actionable yet: {verdict}, edge {edge}%, signals {signals}"


def build_trade_workups(
    candidates: pd.DataFrame,
    *,
    top_n: int,
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
    exclude_tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()
    excluded = {str(t).strip().upper() for t in (exclude_tickers or []) if str(t).strip()}
    ticker_series = df.get("ticker", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    score = pd.to_numeric(df.get("swing_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    edge = pd.to_numeric(df.get("edge_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    signals = pd.to_numeric(df.get("backtest_signals", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    verdict = df.get("backtest_verdict", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()

    mask = ~ticker_series.isin(excluded)
    mask &= score.ge(float(min_swing_score))
    mask &= edge.ge(float(min_edge))
    mask &= signals.ge(int(min_workup_signals))
    mask &= _truthy_mask(df, "quality_gate_pass")
    mask &= _truthy_mask(df, "schwab_gate_pass")
    mask &= _truthy_mask(df, "quote_replay_gate_pass")
    if "strategy_family_gate_pass" in df.columns or "ticker_playbook_gate_pass" in df.columns:
        research_gate = df.apply(research_gate_passes, axis=1)
        mask &= research_gate
    mask &= ~df.apply(_has_hard_quality_reject, axis=1)
    mask &= verdict.eq("LOW_SAMPLE") | (verdict.eq("PASS") & signals.lt(int(min_signals)))

    workups = df[mask].copy()
    if workups.empty:
        return pd.DataFrame()

    workups["_sort_score"] = score.loc[workups.index].fillna(-1)
    workups["_sort_signals"] = signals.loc[workups.index].fillna(0)
    workups["_sort_edge"] = edge.loc[workups.index].fillna(-999)
    workups["workup_reason"] = workups.apply(
        lambda r: _workup_reason(r, min_signals=int(min_signals)),
        axis=1,
    )
    workups["workup_next_step"] = (
        "Build thesis and entry trigger; do not size as Actionable Now until the "
        "sample gate clears or a human explicitly accepts low-sample risk."
    )
    workups = workups.sort_values(
        ["_sort_score", "_sort_signals", "_sort_edge"],
        ascending=[False, False, False],
    )
    workups = _dedupe_trade_rows(workups, by_ticker=True).head(max(1, int(top_n)))
    return workups.drop(
        columns=["_sort_score", "_sort_signals", "_sort_edge"],
        errors="ignore",
    ).reset_index(drop=True)


def _current_setup_tier(row: pd.Series, *, min_signals: int) -> str:
    reasons = str(row.get("actionability_reject_reasons", "") or "").lower()
    verdict = _verdict_text(row.get("backtest_verdict", ""))
    signals = _safe_int(row.get("backtest_signals"))
    if _truthy(row.get("base_gate_pass")) and _truthy(row.get("quality_gate_pass")):
        return "ORDER_READY"
    if (
        "research confidence audit" in reasons
        and _truthy(row.get("quality_gate_pass"))
        and _truthy(row.get("schwab_gate_pass"))
        and _truthy(row.get("quote_replay_gate_pass"))
    ):
        return "RESEARCH_BLOCKED"
    if (
        "strategy family audit" in reasons
        and _truthy(row.get("quality_gate_pass"))
        and _truthy(row.get("schwab_gate_pass"))
        and _truthy(row.get("quote_replay_gate_pass"))
    ):
        return "RESEARCH_BLOCKED"
    if (
        _truthy(row.get("quality_gate_pass"))
        and _truthy(row.get("schwab_gate_pass"))
        and _truthy(row.get("quote_replay_gate_pass"))
        and (verdict == "LOW_SAMPLE" or (verdict == "PASS" and signals < int(min_signals)))
    ):
        return "TRADE_SETUP"
    if (
        _truthy(row.get("base_gate_pass"))
        and _truthy(row.get("schwab_gate_pass"))
        and _truthy(row.get("quote_replay_gate_pass"))
        and not _truthy(row.get("quality_gate_pass"))
        and not _has_hard_quality_reject(row)
    ):
        return "REBUILD"
    return ""


def _entry_trigger(row: pd.Series) -> str:
    strategy = str(row.get("strategy", "") or "").strip().lower()
    direction = str(row.get("direction", "") or "").strip().lower()
    latest = _safe_float(row.get("latest_close"))
    long_strike = _first_finite(row, ("live_long_strike", "long_strike"))
    short_strike = _first_finite(row, ("live_short_strike", "short_strike"))
    width = abs(_safe_float(row.get("spread_width")))
    debit_cap = width * DEFAULT_MAX_DEBIT_TO_WIDTH_PCT if math.isfinite(width) and width > 0 else math.nan

    price_clause = "price confirms the trade direction"
    if direction == "bullish":
        if math.isfinite(long_strike) and math.isfinite(latest) and long_strike > latest:
            otm_pct = (long_strike - latest) / latest if latest > 0 else math.inf
            if math.isfinite(otm_pct) and otm_pct <= DEFAULT_MAX_LONG_STRIKE_OTM_PCT:
                stop_level = latest * 0.99
                price_clause = (
                    f"stock holds above ${stop_level:.2f} and continues pushing toward "
                    f"the long strike ${long_strike:.2f}"
                )
            else:
                price_clause = f"stock reclaims/closes above the long strike ${long_strike:.2f}"
        elif math.isfinite(latest):
            price_clause = f"stock holds above ${latest:.2f} and continues making higher intraday lows"
    elif direction == "bearish":
        if math.isfinite(long_strike) and math.isfinite(latest) and long_strike < latest:
            otm_pct = (latest - long_strike) / latest if latest > 0 else math.inf
            if math.isfinite(otm_pct) and otm_pct <= DEFAULT_MAX_LONG_STRIKE_OTM_PCT:
                stop_level = latest * 1.01
                price_clause = (
                    f"stock stays below ${stop_level:.2f} and continues pushing toward "
                    f"the long put strike ${long_strike:.2f}"
                )
            else:
                price_clause = f"stock loses/closes below the long put strike ${long_strike:.2f}"
        elif math.isfinite(latest):
            price_clause = f"stock stays below ${latest:.2f} and continues making lower intraday highs"
    elif "iron condor" in strategy:
        price_clause = "stock remains inside the short strikes and realized range stays muted"

    cost_clause = ""
    if "debit" in strategy and math.isfinite(debit_cap):
        cost_clause = f"; max debit <= ${debit_cap:.2f}"
    elif "credit" in strategy and math.isfinite(short_strike):
        cost_clause = f"; short strike still has acceptable delta/liquidity near ${short_strike:.2f}"

    return (
        f"Enter only after {price_clause}, both legs quote cleanly, and flow still agrees"
        f"{cost_clause}."
    )


def _current_setup_reason(row: pd.Series, tier: str, *, min_signals: int) -> str:
    if tier == "ORDER_READY":
        return "Full current gate passed; verify quote freshness before any order."
    if tier == "RESEARCH_BLOCKED":
        family_summary = str(row.get("strategy_family_summary", "") or "").strip()
        if family_summary:
            return f"Current tradeability passed, but the matching strategy family is not promotable yet: {family_summary}."
        return "Current tradeability passed, but historical fixed-bucket research is not supportive yet."
    if tier == "TRADE_SETUP":
        return _workup_reason(row, min_signals=int(min_signals))
    if tier == "REBUILD":
        return str(row.get("quality_reject_reasons", "") or "base gates passed, but tradeability needs repair").strip()
    return str(row.get("actionability_reject_reasons", "") or "").strip()


def _current_setup_next_step(row: pd.Series, tier: str) -> str:
    reason = str(row.get("setup_reason", "") or row.get("actionability_reject_reasons", "") or "").lower()
    if tier == "ORDER_READY":
        return "Reprice live, confirm max risk, then use the setup only if the entry trigger is still true."
    if tier == "RESEARCH_BLOCKED":
        return "Paper-track or wait for a promotable strategy-family audit; do not size as Actionable Now."
    if tier == "TRADE_SETUP":
        return "Work the thesis and rerun; this is a conditional setup, not an order ticket."
    if "wide market" in reason:
        return "Rebuild around tighter strikes or wait for bid/ask to tighten before considering entry."
    if "weak directional flow" in reason:
        return "Wait for flow persistence to improve above the directional minimum before using the spread."
    if "earnings" in reason:
        return "Avoid the shown expiry unless the structure is rebuilt fully before earnings."
    return "Keep on watch; fix the listed blocker before considering the setup."


def build_current_trade_setups(
    candidates: pd.DataFrame,
    *,
    top_n: int,
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    df = candidates.copy()
    score = pd.to_numeric(df.get("swing_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    edge = pd.to_numeric(df.get("edge_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    signals = pd.to_numeric(df.get("backtest_signals", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    df["_setup_tier"] = df.apply(lambda r: _current_setup_tier(r, min_signals=int(min_signals)), axis=1)
    tier_rank = {"ORDER_READY": 0, "RESEARCH_BLOCKED": 1, "TRADE_SETUP": 2, "REBUILD": 3}

    mask = df["_setup_tier"].isin(tier_rank)
    mask &= score.ge(float(min_swing_score))
    mask &= ~df.apply(_has_hard_quality_reject, axis=1)
    mask &= (
        _truthy_mask(df, "base_gate_pass")
        | signals.ge(int(min_workup_signals))
        | edge.ge(float(min_edge))
    )
    setups = df[mask].copy()
    if setups.empty:
        return pd.DataFrame()

    setups["setup_tier"] = setups["_setup_tier"]
    setups["setup_reason"] = setups.apply(
        lambda r: _current_setup_reason(r, str(r.get("setup_tier", "")), min_signals=int(min_signals)),
        axis=1,
    )
    setups["setup_entry_trigger"] = setups.apply(_entry_trigger, axis=1)
    setups["setup_next_step"] = setups.apply(
        lambda r: _current_setup_next_step(r, str(r.get("setup_tier", ""))),
        axis=1,
    )
    setups["_tier_rank"] = setups["setup_tier"].map(tier_rank).fillna(99)
    setups["_sort_score"] = score.loc[setups.index].fillna(-1)
    setups["_sort_edge"] = edge.loc[setups.index].fillna(-999)
    setups["_sort_signals"] = signals.loc[setups.index].fillna(0)
    setups = setups.sort_values(
        ["_tier_rank", "_sort_score", "_sort_edge", "_sort_signals"],
        ascending=[True, False, False, False],
    )
    setups = _dedupe_trade_rows(setups, by_ticker=True).head(max(1, int(top_n)))
    return setups.drop(
        columns=["_setup_tier", "_tier_rank", "_sort_score", "_sort_edge", "_sort_signals"],
        errors="ignore",
    ).reset_index(drop=True)


def _max_conviction_reasons(
    row: pd.Series,
    *,
    min_score: float = DEFAULT_MAX_CONVICTION_MIN_SCORE,
    min_edge: float = DEFAULT_MAX_CONVICTION_MIN_EDGE,
    min_signals: int = DEFAULT_MAX_CONVICTION_MIN_SIGNALS,
    min_whale_coverage: float = DEFAULT_MAX_CONVICTION_MIN_WHALE_COVERAGE,
) -> List[str]:
    reasons: List[str] = []
    direction = str(row.get("direction", "") or "").strip().lower()
    score = _safe_float(row.get("swing_score"))
    edge = _safe_float(row.get("edge_pct"))
    signals = _safe_int(row.get("backtest_signals"))
    whales = _safe_int(row.get("whale_appearances"))
    observed = max(1, _safe_int(row.get("days_observed")))

    if not math.isfinite(score) or score < float(min_score):
        reasons.append(f"score {_fmt_num(score, 1)} < max-conviction {float(min_score):.1f}")
    if not math.isfinite(edge) or edge < float(min_edge):
        reasons.append(f"edge {_fmt_num(edge, 1)}% < max-conviction {float(min_edge):.1f}%")
    if signals < int(min_signals):
        reasons.append(f"signals {signals} < max-conviction {int(min_signals)}")
    if whales / observed < float(min_whale_coverage):
        reasons.append(
            f"whale coverage {whales}/{observed} < {float(min_whale_coverage):.0%}"
        )

    if direction in {"bullish", "bearish"}:
        for label, col in (
            ("price", "price_direction"),
            ("flow", "flow_direction"),
            ("OI", "oi_direction"),
        ):
            value = str(row.get(col, "") or "").strip().lower()
            if value != direction:
                reasons.append(f"{label} direction {value or '-'} != {direction}")
        dp = str(row.get("dp_direction", "") or "").strip().lower()
        expected_dp = "accumulation" if direction == "bullish" else "distribution"
        if dp != expected_dp:
            reasons.append(f"DP {dp or '-'} != {expected_dp}")
    else:
        reasons.append("max-conviction tier currently requires a bullish or bearish direction")

    ba_width = _safe_float(row.get("live_bid_ask_width"))
    live_price = abs(_safe_float(row.get("live_spread_cost")))
    spread_width = abs(_safe_float(row.get("spread_width")))
    if math.isfinite(ba_width) and math.isfinite(live_price) and live_price > 0:
        ba_to_price = ba_width / live_price
        if ba_to_price > 0.10:
            reasons.append(f"bid/ask {ba_to_price:.0%} of spread price > max-conviction 10%")
    else:
        reasons.append("missing max-conviction spread price/liquidity")
    if math.isfinite(ba_width) and math.isfinite(spread_width) and spread_width > 0:
        ba_to_width = ba_width / spread_width
        if ba_to_width > 0.05:
            reasons.append(f"bid/ask {ba_to_width:.0%} of spread width > max-conviction 5%")
    else:
        reasons.append("missing max-conviction spread width/liquidity")

    return reasons


def build_max_conviction(actionable: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if actionable.empty:
        return actionable.copy()
    df = actionable.copy()
    df["max_conviction_reasons"] = df.apply(
        lambda r: "; ".join(_max_conviction_reasons(r)),
        axis=1,
    )
    out = df[df["max_conviction_reasons"].astype(str).str.strip().eq("")].copy()
    if out.empty:
        return pd.DataFrame()
    out["position_size_tier"] = "MAX_PLANNED_RISK"
    out["max_planned_risk_units"] = float(MIN_CONFIDENCE_TIERS["MAX_PLANNED_RISK"]["risk_units"])
    out["position_size_guidance"] = str(MIN_CONFIDENCE_TIERS["MAX_PLANNED_RISK"]["guidance"])
    out["max_conviction_instruction"] = (
        "Highest conviction tier: use only the pre-defined max risk budget for one defined-risk trade; not all account capital."
    )
    out = _sort_trade_rows(out)
    out = _dedupe_trade_rows(out, by_ticker=True).head(max(1, int(top_n)))
    return out.reset_index(drop=True)


def _confidence_score(row: pd.Series) -> float:
    score = 0.0
    swing_score = _safe_float(row.get("swing_score"))
    edge = _safe_float(row.get("edge_pct"))
    signals = _safe_int(row.get("backtest_signals"))
    playbook_validation = _safe_int(row.get("ticker_playbook_validation_setups"))
    rolling_verdict = str(row.get("rolling_playbook_verdict", "") or "").strip()

    if math.isfinite(swing_score):
        score += min(25.0, max(0.0, swing_score) * 0.25)
    if math.isfinite(edge):
        score += min(15.0, max(0.0, edge) * 0.5)
    if signals >= DEFAULT_MAX_CONVICTION_MIN_SIGNALS:
        score += 20.0
    elif signals >= DEFAULT_MIN_WORKUP_SIGNALS:
        score += 10.0
    if _truthy(row.get("ticker_playbook_gate_pass")):
        score += 15.0
    if playbook_validation >= TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS:
        score += 5.0
    if rolling_verdict == "supportive":
        score += 15.0
    elif rolling_verdict == "emerging_forward":
        score += 7.0
    if _truthy(row.get("quote_replay_gate_pass")):
        score += 3.0
    if _truthy(row.get("schwab_gate_pass")):
        score += 2.0
    return round(min(100.0, score), 1)


def annotate_position_sizing(actionable: pd.DataFrame) -> pd.DataFrame:
    if actionable.empty:
        return actionable.copy()
    out = actionable.copy()
    tiers: List[str] = []
    scores: List[float] = []
    notes: List[str] = []
    risk_units: List[float] = []
    guidance: List[str] = []
    for _, row in out.iterrows():
        confidence = _confidence_score(row)
        scores.append(confidence)
        signals = _safe_int(row.get("backtest_signals"))
        swing_score = _safe_float(row.get("swing_score"))
        edge = _safe_float(row.get("edge_pct"))
        playbook_validation = _safe_int(row.get("ticker_playbook_validation_setups"))
        rolling_verdict = str(row.get("rolling_playbook_verdict", "") or "").strip()
        verdict = str(row.get("backtest_verdict", "") or "").strip().upper()

        tier = "PROBE_ONLY"
        reason_parts: List[str] = []
        if signals < DEFAULT_MIN_WORKUP_SIGNALS and playbook_validation < TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS:
            reason_parts.append("thin historical sample")
        if rolling_verdict in {"", "no_match", "no_audit", "insufficient_forward"}:
            reason_parts.append("rolling forward validation not proven")
        elif rolling_verdict == "emerging_forward":
            reason_parts.append("rolling forward validation is emerging, not mature")
        elif rolling_verdict == "negative":
            reason_parts.append("rolling forward validation negative")

        if (
            verdict == "PASS"
            and signals >= DEFAULT_MAX_CONVICTION_MIN_SIGNALS
            and math.isfinite(swing_score)
            and swing_score >= DEFAULT_CANDIDATE_MIN_SCORE
            and math.isfinite(edge)
            and edge >= 0
            and rolling_verdict == "supportive"
        ):
            tier = "STANDARD_RISK"
            reason_parts.append("PASS sample plus supportive rolling forward validation")
        elif (
            _truthy(row.get("ticker_playbook_gate_pass"))
            and playbook_validation >= TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS
            and rolling_verdict in {"supportive", "emerging_forward"}
            and math.isfinite(swing_score)
            and swing_score >= DEFAULT_CANDIDATE_MIN_SCORE
        ):
            tier = "STARTER_RISK"
            reason_parts.append("ticker playbook is promotable but still below standard-size evidence")
        elif _truthy(row.get("ticker_playbook_gate_pass")) or signals >= DEFAULT_MIN_WORKUP_SIGNALS:
            tier = "STARTER_RISK"
            reason_parts.append("actionable but below standard-size confidence")

        cfg = MIN_CONFIDENCE_TIERS[tier]
        tiers.append(tier)
        risk_units.append(float(cfg["risk_units"]))
        guidance.append(str(cfg["guidance"]))
        notes.append("; ".join(dict.fromkeys(p for p in reason_parts if p)) or "minimum confidence tier applied")

    out["confidence_score"] = scores
    out["position_size_tier"] = tiers
    out["max_planned_risk_units"] = risk_units
    out["position_size_guidance"] = guidance
    out["position_size_reason"] = notes
    return out


def _spread_width_value(row: pd.Series) -> float:
    width = abs(_safe_float(row.get("spread_width")))
    if math.isfinite(width) and width > 0:
        return width
    long_strike = _first_finite(row, ("live_long_strike", "long_strike"))
    short_strike = _first_finite(row, ("live_short_strike", "short_strike"))
    if math.isfinite(long_strike) and math.isfinite(short_strike):
        return abs(long_strike - short_strike)
    return math.nan


def _entry_price_value(row: pd.Series) -> float:
    price = _first_finite(row, ("live_spread_cost", "quote_replay_entry_net", "est_cost"))
    return abs(price) if math.isfinite(price) else math.nan


def _strategy_cost_type(strategy: str, fallback: str = "") -> str:
    text = str(strategy or "").strip().lower()
    if "credit" in text or "condor" in text:
        return "credit"
    if "debit" in text:
        return "debit"
    fallback_text = str(fallback or "").strip().lower()
    if fallback_text in {"credit", "debit"}:
        return fallback_text
    return fallback_text


def _parse_tracker_vertical_legs(row: pd.Series) -> Dict[str, Any]:
    long_strike = _first_finite(row, ("long_strike", "live_long_strike"))
    short_strike = _first_finite(row, ("short_strike", "live_short_strike"))
    setup = str(row.get("trade_setup", "") or row.get("strike_setup", "") or "").strip()
    matches = re.findall(r"\b(Buy|Sell)\s+([0-9]+(?:\.[0-9]+)?)([CP])\b", setup, flags=re.IGNORECASE)
    right = ""
    if matches:
        for action, strike, option_right in matches:
            right = option_right.upper()
            parsed = _safe_float(strike)
            if not math.isfinite(parsed):
                continue
            if action.lower() == "buy" and not math.isfinite(long_strike):
                long_strike = parsed
            elif action.lower() == "sell" and not math.isfinite(short_strike):
                short_strike = parsed
    return {
        "long_strike": long_strike,
        "short_strike": short_strike,
        "right": right,
    }


def _tracker_row_to_candidate(row: pd.Series) -> Optional[Dict[str, Any]]:
    signal_date = _parse_tracker_date(row.get("signal_date"))
    expiry = _parse_tracker_date(row.get("target_expiry"))
    ticker = str(row.get("ticker", "") or "").strip().upper()
    strategy = str(row.get("strategy", "") or "").strip()
    if signal_date is None or expiry is None or not ticker or not strategy:
        return None
    legs = _parse_tracker_vertical_legs(row)
    long_strike = _safe_float(legs.get("long_strike"))
    short_strike = _safe_float(legs.get("short_strike"))
    if not math.isfinite(long_strike) or not math.isfinite(short_strike):
        return None
    width = abs(short_strike - long_strike)
    if width <= 0:
        return None
    return {
        "ticker": ticker,
        "strategy": strategy,
        "target_expiry": expiry.isoformat(),
        "long_strike": long_strike,
        "short_strike": short_strike,
        "spread_width": width,
        "cost_type": _strategy_cost_type(strategy, str(row.get("cost_type", "") or "")),
        "live_validated": False,
    }


def _trade_tracking_id(as_of: dt.date, row: pd.Series) -> str:
    parts = [
        as_of.isoformat(),
        str(row.get("ticker", "") or "").strip().upper(),
        str(row.get("direction", "") or "").strip().lower(),
        str(row.get("strategy", "") or "").strip(),
        str(row.get("target_expiry", "") or "").strip(),
        _entry_text(row),
    ]
    return "|".join(parts)


def update_trade_tracking(
    actionable: pd.DataFrame,
    tracking_csv: Path,
    *,
    report_path: Path,
    as_of: dt.date,
    enabled: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "tracking_csv": str(tracking_csv),
        "added": 0,
        "updated": 0,
        "total": 0,
    }
    if not enabled:
        summary["status"] = "skipped"
        return summary
    tracking_csv.parent.mkdir(parents=True, exist_ok=True)
    if tracking_csv.exists():
        try:
            tracker = pd.read_csv(tracking_csv, low_memory=False)
        except Exception:
            tracker = pd.DataFrame(columns=TRACKING_COLUMNS)
    else:
        tracker = pd.DataFrame(columns=TRACKING_COLUMNS)
    for col in TRACKING_COLUMNS:
        if col not in tracker.columns:
            tracker[col] = ""
        tracker[col] = tracker[col].astype(object)
        tracker[col] = tracker[col].astype(object)
    tracker["trade_id"] = tracker["trade_id"].fillna("").astype(str)
    existing_ids = set(tracker["trade_id"].tolist())
    added_rows: List[Dict[str, Any]] = []
    updated = 0
    if not actionable.empty:
        for _, row in actionable.iterrows():
            trade_id = _trade_tracking_id(as_of, row)
            if trade_id in existing_ids:
                tracker.loc[tracker["trade_id"].eq(trade_id), "last_seen_as_of"] = as_of.isoformat()
                updated += 1
                continue
            entry_price = _entry_price_value(row)
            width = _spread_width_value(row)
            cost_type = str(row.get("cost_type", "") or "").strip().lower()
            long_strike = _first_finite(row, ("live_long_strike", "long_strike"))
            short_strike = _first_finite(row, ("live_short_strike", "short_strike"))
            max_risk = math.nan
            max_profit = math.nan
            if math.isfinite(entry_price) and math.isfinite(width) and width > 0:
                if "credit" in cost_type:
                    max_risk = max(0.0, (width - entry_price) * 100.0)
                    max_profit = entry_price * 100.0
                else:
                    max_risk = entry_price * 100.0
                    max_profit = max(0.0, (width - entry_price) * 100.0)
            added_rows.append(
                {
                    "trade_id": trade_id,
                    "status": "OPEN_TRACKED",
                    "signal_date": as_of.isoformat(),
                    "last_seen_as_of": as_of.isoformat(),
                    "ticker": str(row.get("ticker", "") or "").strip().upper(),
                    "direction": str(row.get("direction", "") or "").strip().lower(),
                    "strategy": str(row.get("strategy", "") or "").strip(),
                    "variant_tag": str(row.get("variant_tag", "") or "base").strip() or "base",
                    "trade_setup": _trade_setup_text(row),
                    "target_expiry": str(row.get("target_expiry", "") or "").strip(),
                    "long_strike": long_strike,
                    "short_strike": short_strike,
                    "spread_width": width,
                    "cost_type": cost_type,
                    "entry_price": entry_price,
                    "max_risk": max_risk,
                    "max_profit": max_profit,
                    "position_size_tier": str(row.get("position_size_tier", "") or "").strip(),
                    "position_size_guidance": str(row.get("position_size_guidance", "") or "").strip(),
                    "entry_trigger": str(row.get("setup_entry_trigger", "") or "").strip() or _entry_trigger(row),
                    "source_report": str(report_path),
                }
            )
            existing_ids.add(trade_id)
    if added_rows:
        added_df = pd.DataFrame(added_rows)
        tracker = added_df if tracker.empty else pd.concat([tracker, added_df], ignore_index=True)
    for col in TRACKING_COLUMNS:
        if col not in tracker.columns:
            tracker[col] = ""
        tracker[col] = tracker[col].astype(object)
    tracker = tracker[TRACKING_COLUMNS]
    tracker.to_csv(tracking_csv, index=False)
    summary.update(
        {
            "status": "ok",
            "added": int(len(added_rows)),
            "updated": int(updated),
            "total": int(len(tracker)),
        }
    )
    return summary


def refresh_trade_tracking_outcomes(
    tracking_csv: Path,
    *,
    root: Path,
    as_of: dt.date,
    enabled: bool,
    replay_fn: Any = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "tracking_csv": str(tracking_csv),
        "updated": 0,
        "scored": 0,
        "unavailable": 0,
        "wins": 0,
        "losses": 0,
        "open": 0,
        "total": 0,
    }
    if not enabled:
        summary["status"] = "skipped"
        return summary
    if not tracking_csv.exists():
        summary["status"] = "missing_tracker"
        return summary
    tracker = pd.read_csv(tracking_csv, low_memory=False)
    for col in TRACKING_COLUMNS:
        if col not in tracker.columns:
            tracker[col] = ""
        tracker[col] = tracker[col].astype(object)
    if tracker.empty:
        tracker.to_csv(tracking_csv, index=False)
        summary["status"] = "empty"
        return summary

    replay = replay_fn or trend_quote_replay.annotate_quote_replay
    candidates_by_signal_date: Dict[dt.date, List[Tuple[int, Dict[str, Any]]]] = {}
    unavailable = 0
    for idx, row in tracker.iterrows():
        signal_date = _parse_tracker_date(row.get("signal_date"))
        if signal_date is None or signal_date > as_of:
            unavailable += 1
            continue
        candidate = _tracker_row_to_candidate(row)
        if candidate is None:
            tracker.at[idx, "outcome_as_of"] = as_of.isoformat()
            tracker.at[idx, "outcome_status"] = "UNAVAILABLE"
            tracker.at[idx, "outcome_verdict"] = "UNAVAILABLE"
            tracker.at[idx, "outcome_reason"] = "could not rebuild tracked option spread"
            unavailable += 1
            continue
        candidates_by_signal_date.setdefault(signal_date, []).append((idx, candidate))

    updated = 0
    scored = 0
    wins = 0
    losses = 0
    open_count = 0
    for signal_date, items in candidates_by_signal_date.items():
        indexes = [idx for idx, _ in items]
        candidates = pd.DataFrame([candidate for _, candidate in items])
        try:
            annotated, _ = replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="diagnostic",
                exit_date_override=as_of,
                allow_web_fallback=False,
            )
        except Exception as exc:
            for idx in indexes:
                tracker.at[idx, "outcome_as_of"] = as_of.isoformat()
                tracker.at[idx, "outcome_status"] = "UNAVAILABLE"
                tracker.at[idx, "outcome_verdict"] = "UNAVAILABLE"
                tracker.at[idx, "outcome_reason"] = f"outcome replay failed: {exc}"
                unavailable += 1
            continue

        for row_pos, idx in enumerate(indexes):
            replay_row = annotated.iloc[row_pos]
            exit_net = _safe_float(replay_row.get("quote_replay_exit_net"))
            entry_price = _safe_float(tracker.at[idx, "entry_price"])
            if not math.isfinite(entry_price) or entry_price <= 0:
                entry_price = _safe_float(replay_row.get("quote_replay_entry_net"))
            width = _safe_float(tracker.at[idx, "spread_width"])
            if not math.isfinite(width) or width <= 0:
                width = _safe_float(replay_row.get("spread_width"))
            net_type = _strategy_cost_type(
                str(tracker.at[idx, "strategy"] or ""),
                str(tracker.at[idx, "cost_type"] or ""),
            )
            final = _truthy(replay_row.get("quote_replay_final"))
            replay_status = str(replay_row.get("quote_replay_status", "") or "").strip()
            replay_reason = str(replay_row.get("quote_replay_reason", "") or "").strip()
            outcome_status = "UNAVAILABLE"
            outcome_verdict = "UNAVAILABLE"
            pnl = math.nan
            return_on_risk = math.nan
            if math.isfinite(exit_net) and math.isfinite(entry_price) and entry_price > 0:
                if net_type == "credit":
                    pnl = (entry_price - exit_net) * 100.0
                else:
                    pnl = (exit_net - entry_price) * 100.0
                max_loss = _safe_float(tracker.at[idx, "max_risk"])
                if not math.isfinite(max_loss) or max_loss <= 0:
                    if net_type == "credit":
                        max_loss = max(0.0, (width - entry_price) * 100.0)
                    else:
                        max_loss = entry_price * 100.0
                return_on_risk = pnl / max_loss if math.isfinite(max_loss) and max_loss > 0 else math.nan
                if final:
                    outcome_status = "CLOSED_WIN" if pnl > 0 else "CLOSED_LOSS"
                    outcome_verdict = "WIN" if pnl > 0 else "LOSS"
                elif pnl > 0:
                    outcome_status = "OPEN_WIN"
                    outcome_verdict = "PARTIAL_WIN"
                elif pnl < 0:
                    outcome_status = "OPEN_LOSS"
                    outcome_verdict = "PARTIAL_LOSS"
                else:
                    outcome_status = "OPEN_FLAT"
                    outcome_verdict = "FLAT"
                scored += 1
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                if not final:
                    open_count += 1
            elif replay_status == "entry_only_no_later_snapshot":
                outcome_status = "OPEN_TRACKED"
                outcome_verdict = "ENTRY_OK"
                open_count += 1
            else:
                unavailable += 1

            tracker.at[idx, "status"] = outcome_status if outcome_status != "UNAVAILABLE" else tracker.at[idx, "status"]
            tracker.at[idx, "outcome_as_of"] = as_of.isoformat()
            tracker.at[idx, "outcome_exit_date"] = str(replay_row.get("quote_replay_exit_date", "") or "")
            tracker.at[idx, "outcome_final"] = bool(final)
            tracker.at[idx, "outcome_status"] = outcome_status
            tracker.at[idx, "outcome_verdict"] = outcome_verdict
            tracker.at[idx, "outcome_entry_price"] = entry_price
            tracker.at[idx, "outcome_exit_price"] = exit_net
            tracker.at[idx, "outcome_pnl"] = pnl
            tracker.at[idx, "outcome_return_on_risk"] = return_on_risk
            tracker.at[idx, "outcome_days_held"] = _safe_int(replay_row.get("quote_replay_days_held"))
            tracker.at[idx, "outcome_reason"] = replay_reason
            tracker.at[idx, "outcome_source"] = "local_uw_option_replay"
            updated += 1

    tracker = tracker[TRACKING_COLUMNS]
    tracker.to_csv(tracking_csv, index=False)
    summary.update(
        {
            "status": "ok",
            "updated": int(updated),
            "scored": int(scored),
            "unavailable": int(unavailable),
            "wins": int(wins),
            "losses": int(losses),
            "open": int(open_count),
            "total": int(len(tracker)),
        }
    )
    return summary


def _parse_walk_forward_horizons(value: str) -> List[int]:
    horizons: List[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        parsed = int(text)
        if parsed <= 0:
            raise ValueError(f"Walk-forward horizon must be positive: {text}")
        horizons.append(parsed)
    return sorted(set(horizons))


def _walk_forward_signal_dates(
    all_days: List[Tuple[dt.date, Path]],
    *,
    lookback: int,
    as_of: dt.date,
    samples: int,
    max_horizon: int,
) -> List[dt.date]:
    if samples <= 0 or max_horizon <= 0:
        return []
    eligible: List[dt.date] = []
    for idx, (day, _) in enumerate(all_days):
        if day >= as_of:
            continue
        if idx < max(0, int(lookback) - 1):
            continue
        if idx + int(max_horizon) >= len(all_days):
            continue
        eligible.append(day)
    return eligible[-int(samples):]


def _signal_dates_by_horizon(
    all_days: List[Tuple[dt.date, Path]],
    *,
    lookback: int,
    as_of: dt.date,
    samples: int,
    horizons: Sequence[int],
) -> Dict[int, List[dt.date]]:
    return {
        int(horizon): _walk_forward_signal_dates(
            all_days,
            lookback=lookback,
            as_of=as_of,
            samples=samples,
            max_horizon=int(horizon),
        )
        for horizon in sorted(set(int(h) for h in horizons if int(h) > 0))
    }


def _market_day_after(
    all_days: List[Tuple[dt.date, Path]],
    signal_date: dt.date,
    horizon: int,
) -> Optional[dt.date]:
    dates = [day for day, _ in all_days]
    try:
        idx = dates.index(signal_date)
    except ValueError:
        return None
    target_idx = idx + int(horizon)
    if target_idx >= len(dates):
        return None
    return dates[target_idx]


def _historical_entry_available(row: pd.Series) -> bool:
    entry = _safe_float(row.get("quote_replay_entry_net"))
    if not math.isfinite(entry) or entry <= 0:
        return False
    status = str(row.get("quote_replay_status", "") or "").strip().lower()
    return status not in {
        "unsupported_strategy",
        "invalid_expiry",
        "invalid_setup",
        "skipped_missing_entry",
    }


def _walk_forward_outcome_rows(
    *,
    selected: pd.DataFrame,
    annotated_by_horizon: Dict[int, pd.DataFrame],
    signal_date: dt.date,
    exit_dates: Dict[int, dt.date],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if selected.empty:
        return rows
    selected_ids = selected.get("_audit_candidate_id", pd.Series(dtype=int)).tolist()
    selected_rank = {
        int(candidate_id): rank
        for rank, candidate_id in enumerate(selected_ids, start=1)
        if math.isfinite(_safe_float(candidate_id))
    }
    for horizon, annotated in annotated_by_horizon.items():
        if annotated.empty or "_audit_candidate_id" not in annotated.columns:
            continue
        by_id = annotated.set_index("_audit_candidate_id", drop=False)
        for candidate_id in selected_ids:
            try:
                candidate_int = int(candidate_id)
            except Exception:
                continue
            if candidate_int not in by_id.index:
                continue
            row = by_id.loc[candidate_int]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pnl = _safe_float(row.get("quote_replay_pnl"))
            ret = _safe_float(row.get("quote_replay_return_on_risk"))
            rows.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "horizon_market_days": int(horizon),
                    "exit_date": (exit_dates.get(horizon) or dt.date.min).isoformat()
                    if exit_dates.get(horizon)
                    else "",
                    "selected_rank": selected_rank.get(candidate_int, 0),
                    "ticker": str(row.get("ticker", "") or "").upper().strip(),
                    "direction": str(row.get("direction", "") or "").strip().lower(),
                    "strategy": str(row.get("strategy", "") or "").strip(),
                    "trade_setup": _trade_setup_text(row),
                    "swing_score": _safe_float(row.get("swing_score")),
                    "edge_pct": _safe_float(row.get("edge_pct")),
                    "backtest_signals": _safe_int(row.get("backtest_signals")),
                    "backtest_verdict": _verdict_text(row.get("backtest_verdict", "")),
                    "entry_available": _historical_entry_available(row),
                    "entry_net": _safe_float(row.get("quote_replay_entry_net")),
                    "exit_net": _safe_float(row.get("quote_replay_exit_net")),
                    "pnl": pnl,
                    "return_on_risk": ret,
                    "win": bool(math.isfinite(pnl) and pnl > 0),
                    "outcome_verdict": str(row.get("quote_replay_verdict", "") or "UNAVAILABLE").strip().upper(),
                    "outcome_status": str(row.get("quote_replay_status", "") or "").strip(),
                    "outcome_reason": str(row.get("quote_replay_reason", "") or "").strip(),
                }
            )
    return rows


def run_walk_forward_audit(
    *,
    root: Path,
    out_dir: Path,
    cfg_template: Dict[str, Any],
    as_of: dt.date,
    lookback: int,
    candidate_pool: int,
    top_n: int,
    samples: int,
    horizons: List[int],
    cache_dir: str,
    min_edge: float,
    min_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
    reuse_raw: bool = False,
) -> pd.DataFrame:
    if samples <= 0 or not horizons:
        return pd.DataFrame(columns=WALK_FORWARD_COLUMNS)

    all_days = swing.discover_trading_days(root, 10000, as_of)
    audit_dates_by_horizon = _signal_dates_by_horizon(
        all_days,
        lookback=lookback,
        as_of=as_of,
        samples=samples,
        horizons=horizons,
    )
    audit_dates = sorted(
        {day for dates in audit_dates_by_horizon.values() for day in dates}
    )
    if not audit_dates:
        return pd.DataFrame(columns=WALK_FORWARD_COLUMNS)

    rows: List[Dict[str, Any]] = []
    audit_root = out_dir / "walk_forward"
    audit_root.mkdir(parents=True, exist_ok=True)
    quote_store = trend_quote_replay.HistoricalOptionQuoteStore(root_dir=root, use_hot=True, use_oi=True)
    close_store = trend_quote_replay.UnderlyingCloseStore(root_dir=root, allow_web_fallback=False)
    print(
        f"  Walk-forward audit: {len(audit_dates)} signal dates, horizons {','.join(map(str, horizons))} market days",
        flush=True,
    )

    for signal_date in audit_dates:
        print(f"    [walk-forward] {signal_date.isoformat()}", flush=True)
        names = _output_names(signal_date, lookback)
        audit_out = audit_root / f"{signal_date.isoformat()}-L{lookback}"
        audit_out.mkdir(parents=True, exist_ok=True)
        raw_csv = audit_out / names["raw_csv"]

        if reuse_raw and raw_csv.exists():
            print(f"      reuse raw candidates: {raw_csv}", flush=True)
        else:
            audit_cfg = copy.deepcopy(cfg_template)
            audit_cfg.setdefault("pipeline", {})["root_dir"] = str(root)
            audit_cfg.setdefault("pipeline", {})["lookback_days"] = lookback
            audit_cfg.setdefault("pipeline", {})["output_dir"] = str(audit_out)
            audit_cfg.setdefault("output", {})["report_md_name"] = names["raw_report"].replace(f"-L{lookback}.md", ".md")
            audit_cfg.setdefault("output", {})["shortlist_csv_name"] = names["raw_csv"].replace(f"-L{lookback}.csv", ".csv")
            audit_cfg.setdefault("schwab_validation", {})["enabled"] = False
            audit_cfg.setdefault("backtest", {})["enabled"] = True
            audit_cfg.setdefault("backtest", {})["min_signals"] = int(min_signals)
            if cache_dir:
                audit_cfg.setdefault("backtest", {})["cache_dir"] = cache_dir
            swing.run_pipeline(
                cfg=audit_cfg,
                root=root,
                lookback=lookback,
                as_of=signal_date,
                out_dir=audit_out,
                max_recommendations=candidate_pool,
            )
        candidates = pd.read_csv(raw_csv, low_memory=False) if raw_csv.exists() else pd.DataFrame()

        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["_audit_candidate_id"] = range(len(candidates))

        annotated_by_horizon: Dict[int, pd.DataFrame] = {}
        exit_dates: Dict[int, dt.date] = {}
        for horizon in horizons:
            if signal_date not in set(audit_dates_by_horizon.get(int(horizon), [])):
                continue
            exit_date = _market_day_after(all_days, signal_date, horizon)
            if exit_date is None:
                continue
            exit_dates[int(horizon)] = exit_date
            annotated, _ = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="diagnostic",
                exit_date_override=exit_date,
                quote_store=quote_store,
                close_store=close_store,
            )
            annotated["historical_entry_gate_pass"] = annotated.apply(_historical_entry_available, axis=1)
            entry_net = pd.to_numeric(annotated.get("quote_replay_entry_net", pd.Series(np.nan, index=annotated.index)), errors="coerce")
            if "live_spread_cost" not in annotated.columns:
                annotated["live_spread_cost"] = np.nan
            live_cost = pd.to_numeric(annotated["live_spread_cost"], errors="coerce")
            annotated["live_spread_cost"] = live_cost.where(live_cost.notna(), entry_net)
            annotated_by_horizon[int(horizon)] = annotated

        if not annotated_by_horizon:
            continue
        selection_source = annotated_by_horizon[min(annotated_by_horizon)].copy()
        actionable, _ = split_actionable_candidates(
            selection_source,
            top_n=max(1, int(top_n)),
            backtest_enabled=True,
            schwab_enabled=False,
            quote_replay_mode="off",
            min_edge=float(min_edge),
            min_signals=int(min_signals),
            min_swing_score=float(min_swing_score),
            allow_low_sample=bool(allow_low_sample),
            allow_earnings_risk=bool(allow_earnings_risk),
            allow_volatile_ic=bool(allow_volatile_ic),
            allow_flow_conflict=bool(allow_flow_conflict),
            max_bid_ask_to_price_pct=float(max_bid_ask_to_price_pct),
            max_bid_ask_to_width_pct=float(max_bid_ask_to_width_pct),
            max_short_delta=float(max_short_delta),
            min_underlying_price=float(min_underlying_price),
            min_debit_spread_price=float(min_debit_spread_price),
            min_whale_appearances=int(min_whale_appearances),
        )
        if not actionable.empty and "historical_entry_gate_pass" in actionable.columns:
            actionable = actionable[actionable["historical_entry_gate_pass"].apply(_truthy)].copy()
        if actionable.empty:
            continue
        rows.extend(
            _walk_forward_outcome_rows(
                selected=actionable.head(max(1, int(top_n))),
                annotated_by_horizon=annotated_by_horizon,
                signal_date=signal_date,
                exit_dates=exit_dates,
            )
        )

    return pd.DataFrame(rows, columns=WALK_FORWARD_COLUMNS)


def _walk_forward_completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pnl" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["_pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    return out[out["_pnl"].notna()].copy()


def _walk_forward_confidence_text(df: pd.DataFrame) -> str:
    completed = _walk_forward_completed(df)
    if completed.empty:
        return "not enough completed historical outcomes yet"
    wins = completed["_pnl"].gt(0)
    hit_rate = float(wins.mean())
    avg_pnl = float(completed["_pnl"].mean())
    ret = pd.to_numeric(completed.get("return_on_risk", pd.Series(np.nan, index=completed.index)), errors="coerce")
    avg_ret = float(ret.mean()) if ret.notna().any() else math.nan
    n = len(completed)
    if n < 10:
        prefix = "low sample"
    elif hit_rate >= 0.60 and avg_pnl > 0 and (not math.isfinite(avg_ret) or avg_ret > 0):
        prefix = "supportive"
    elif hit_rate >= 0.50 and avg_pnl >= 0:
        prefix = "mixed/supportive"
    else:
        prefix = "negative"
    ret_text = f", avg return on risk {avg_ret:.1%}" if math.isfinite(avg_ret) else ""
    return f"{prefix}: {n} outcomes, hit rate {hit_rate:.0%}, avg P&L ${avg_pnl:.0f}{ret_text}"


def _walk_forward_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no historical candidate cleared the audit selection gates with completed future option quotes yet_"]
    lines = [
        "This is an anti-overfit check: historical trades are selected using signal-date evidence only; future option quotes are used only to score the outcome.",
        "",
        f"- **Confidence impact:** {_walk_forward_confidence_text(df)}",
    ]
    completed = _walk_forward_completed(df)
    if completed.empty:
        return lines
    grouped = completed.groupby("horizon_market_days", dropna=False)
    for horizon, group in grouped:
        pnl = pd.to_numeric(group["pnl"], errors="coerce")
        ret = pd.to_numeric(group.get("return_on_risk", pd.Series(np.nan, index=group.index)), errors="coerce")
        wins = pnl.gt(0)
        ret_text = f"; avg R/R {float(ret.mean()):.1%}" if ret.notna().any() else ""
        lines.append(
            f"- **{int(horizon)} market days:** {len(group)} outcomes; "
            f"hit {float(wins.mean()):.0%}; avg P&L ${float(pnl.mean()):.0f}{ret_text}"
        )
    worst = completed.sort_values("_pnl").head(3)
    if not worst.empty:
        parts = [
            f"{str(r.get('ticker', '')).upper()} {int(r.get('horizon_market_days', 0))}d ${float(r.get('_pnl')):.0f}"
            for _, r in worst.iterrows()
        ]
        lines.append(f"- **Worst false positives:** {', '.join(parts)}")
    return lines


def _research_audit_bucket_verdict(
    *,
    outcomes: int,
    unique_setups: int,
    hit_rate: float,
    avg_pnl: float,
    avg_return_on_risk: float,
) -> str:
    if outcomes <= 0:
        return "empty"
    if outcomes < 10 or unique_setups < 5:
        return "low_sample"
    if math.isfinite(avg_pnl) and avg_pnl < 0:
        return "negative"
    if (
        outcomes >= RESEARCH_AUDIT_MIN_OUTCOMES
        and unique_setups >= RESEARCH_AUDIT_MIN_UNIQUE_SETUPS
        and math.isfinite(hit_rate)
        and hit_rate >= 0.55
        and math.isfinite(avg_pnl)
        and avg_pnl > 0
        and (not math.isfinite(avg_return_on_risk) or avg_return_on_risk > 0)
    ):
        return "supportive"
    if unique_setups < RESEARCH_AUDIT_MIN_UNIQUE_SETUPS:
        return "low_sample"
    return "mixed"


def _research_policy_frames(
    annotated: pd.DataFrame,
    *,
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
) -> Dict[str, pd.DataFrame]:
    if annotated.empty:
        return {}
    df = annotated.copy()
    df["base_gate_reasons"] = df.apply(
        lambda r: "; ".join(
            base_gate_reasons(
                r,
                backtest_enabled=True,
                schwab_enabled=False,
                quote_replay_mode="off",
                min_edge=min_edge,
                min_signals=min_signals,
                min_swing_score=min_swing_score,
                allow_low_sample=allow_low_sample,
            )
        ),
        axis=1,
    )
    df["quality_reject_reasons"] = df.apply(
        lambda r: "; ".join(
            quality_gate_reasons(
                r,
                schwab_enabled=False,
                allow_earnings_risk=allow_earnings_risk,
                allow_volatile_ic=allow_volatile_ic,
                allow_flow_conflict=allow_flow_conflict,
                max_bid_ask_to_price_pct=max_bid_ask_to_price_pct,
                max_bid_ask_to_width_pct=max_bid_ask_to_width_pct,
                max_short_delta=max_short_delta,
                min_underlying_price=min_underlying_price,
                min_debit_spread_price=min_debit_spread_price,
                min_whale_appearances=min_whale_appearances,
            )
        ),
        axis=1,
    )
    df["_base_ok"] = df["base_gate_reasons"].fillna("").astype(str).str.strip().eq("")
    df["_quality_ok"] = df["quality_reject_reasons"].fillna("").astype(str).str.strip().eq("")
    df["_entry_ok"] = df.apply(_historical_entry_available, axis=1)
    pnl = pd.to_numeric(df.get("quote_replay_pnl", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["_completed"] = pnl.notna()
    score = pd.to_numeric(df.get("swing_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    signals = pd.to_numeric(df.get("backtest_signals", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    verdict = df.get("backtest_verdict", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    edge = pd.to_numeric(df.get("edge_pct", pd.Series(np.nan, index=df.index)), errors="coerce")

    base_mask = df["_entry_ok"] & df["_completed"] & score.ge(float(min_swing_score))
    workup_mask = (
        base_mask
        & df["_quality_ok"]
        & verdict.eq("LOW_SAMPLE")
        & edge.ge(float(min_edge))
        & signals.ge(int(min_workup_signals))
    )
    frames = {
        "entry_available_score_gate": df[base_mask].copy(),
        "backtest_pass_sample_gate": df[base_mask & df["_base_ok"]].copy(),
        "professional_quality_gate": df[base_mask & df["_quality_ok"]].copy(),
        "backtest_plus_professional_gate": df[base_mask & df["_base_ok"] & df["_quality_ok"]].copy(),
        "trade_workup_quality_gate": df[workup_mask].copy(),
    }
    return {k: _dedupe_trade_rows(v) for k, v in frames.items()}


def _research_unique_setup_count(completed: pd.DataFrame) -> int:
    unique_cols = [c for c in ["signal_date", "ticker", "trade_setup"] if c in completed.columns]
    if not unique_cols:
        return int(len(completed))
    return int(completed[unique_cols].drop_duplicates().shape[0])


def _research_summary_row(group: pd.DataFrame, *, policy: str, horizon: Optional[int] = None) -> Optional[Dict[str, Any]]:
    pnl_all = pd.to_numeric(group.get("pnl", pd.Series(np.nan, index=group.index)), errors="coerce")
    completed_mask = pnl_all.notna()
    if not completed_mask.any():
        return None
    completed = group[completed_mask].copy()
    pnl = pnl_all[completed_mask]
    ret = pd.to_numeric(
        completed.get("return_on_risk", pd.Series(np.nan, index=completed.index)),
        errors="coerce",
    )
    unique_setups = _research_unique_setup_count(completed)
    avg_horizons = float(len(pnl) / unique_setups) if unique_setups > 0 else math.nan
    hit_rate = float(pnl.gt(0).mean())
    avg_pnl = float(pnl.mean())
    median_pnl = float(pnl.median())
    avg_ret = float(ret.mean()) if ret.notna().any() else math.nan
    row: Dict[str, Any] = {
        "policy": str(policy),
        "outcomes": int(len(pnl)),
        "unique_setups": unique_setups,
        "hit_rate": hit_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "avg_return_on_risk": avg_ret,
        "worst_pnl": float(pnl.min()),
        "verdict": _research_audit_bucket_verdict(
            outcomes=int(len(pnl)),
            unique_setups=unique_setups,
            hit_rate=hit_rate,
            avg_pnl=avg_pnl,
            avg_return_on_risk=avg_ret,
        ),
    }
    if horizon is None:
        row["avg_horizons_per_setup"] = avg_horizons
    else:
        row["horizon_market_days"] = int(horizon)
    return row


def _research_summary_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=RESEARCH_AUDIT_COLUMNS)
    rows: List[Dict[str, Any]] = []
    for policy, group in outcomes.groupby("policy", dropna=False):
        row = _research_summary_row(group, policy=str(policy))
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=RESEARCH_AUDIT_COLUMNS)
    return pd.DataFrame(rows, columns=RESEARCH_AUDIT_COLUMNS)


def _research_summary_by_horizon_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty or "horizon_market_days" not in outcomes.columns:
        return pd.DataFrame(columns=RESEARCH_HORIZON_AUDIT_COLUMNS)
    rows: List[Dict[str, Any]] = []
    for (policy, horizon), group in outcomes.groupby(["policy", "horizon_market_days"], dropna=False):
        parsed_horizon = _safe_int(horizon)
        if parsed_horizon <= 0:
            continue
        row = _research_summary_row(group, policy=str(policy), horizon=parsed_horizon)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=RESEARCH_HORIZON_AUDIT_COLUMNS)
    return pd.DataFrame(rows, columns=RESEARCH_HORIZON_AUDIT_COLUMNS)


def _research_confidence_supportive(
    df: pd.DataFrame,
    horizon_df: Optional[pd.DataFrame] = None,
) -> bool:
    if horizon_df is not None and not horizon_df.empty and "verdict" in horizon_df.columns:
        verdicts = horizon_df["verdict"].fillna("").astype(str).str.lower()
        return bool(verdicts.eq("supportive").any())
    if df.empty or "verdict" not in df.columns:
        return False
    verdicts = df["verdict"].fillna("").astype(str).str.lower()
    return bool(verdicts.eq("supportive").any())


def collect_research_confidence_outcomes(
    *,
    root: Path,
    out_dir: Path,
    as_of: dt.date,
    lookback: int,
    samples: int,
    horizons: List[int],
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
) -> pd.DataFrame:
    if samples <= 0 or not horizons:
        return pd.DataFrame(columns=RESEARCH_OUTCOME_COLUMNS)
    all_days = swing.discover_trading_days(root, 10000, as_of)
    audit_dates_by_horizon = _signal_dates_by_horizon(
        all_days,
        lookback=lookback,
        as_of=as_of,
        samples=samples,
        horizons=horizons,
    )
    if not any(audit_dates_by_horizon.values()):
        return pd.DataFrame(columns=RESEARCH_OUTCOME_COLUMNS)

    outcome_rows: List[Dict[str, Any]] = []
    audit_root = out_dir / "walk_forward"
    quote_store = trend_quote_replay.HistoricalOptionQuoteStore(root_dir=root, use_hot=True, use_oi=True)
    close_store = trend_quote_replay.UnderlyingCloseStore(root_dir=root, allow_web_fallback=False)
    for horizon in horizons:
        for signal_date in audit_dates_by_horizon.get(int(horizon), []):
            names = _output_names(signal_date, lookback)
            raw_csv = audit_root / f"{signal_date.isoformat()}-L{lookback}" / names["raw_csv"]
            if not raw_csv.exists():
                continue
            raw = pd.read_csv(raw_csv, low_memory=False)
            if raw.empty:
                continue
            raw = raw.copy()
            score = pd.to_numeric(raw.get("swing_score", pd.Series(np.nan, index=raw.index)), errors="coerce")
            raw = raw[score.ge(float(min_swing_score))].copy()
            if raw.empty:
                continue
            exit_date = _market_day_after(all_days, signal_date, int(horizon))
            if exit_date is None:
                continue
            annotated, _ = trend_quote_replay.annotate_quote_replay(
                raw,
                root=root,
                signal_date=signal_date,
                mode="diagnostic",
                exit_date_override=exit_date,
                quote_store=quote_store,
                close_store=close_store,
            )
            entry_net = pd.to_numeric(
                annotated.get("quote_replay_entry_net", pd.Series(np.nan, index=annotated.index)),
                errors="coerce",
            )
            if "live_spread_cost" not in annotated.columns:
                annotated["live_spread_cost"] = np.nan
            live_cost = pd.to_numeric(annotated["live_spread_cost"], errors="coerce")
            annotated["live_spread_cost"] = live_cost.where(live_cost.notna(), entry_net)
            policy_frames = _research_policy_frames(
                annotated,
                min_edge=min_edge,
                min_signals=min_signals,
                min_workup_signals=min_workup_signals,
                min_swing_score=min_swing_score,
                allow_low_sample=allow_low_sample,
                allow_earnings_risk=allow_earnings_risk,
                allow_volatile_ic=allow_volatile_ic,
                allow_flow_conflict=allow_flow_conflict,
                max_bid_ask_to_price_pct=max_bid_ask_to_price_pct,
                max_bid_ask_to_width_pct=max_bid_ask_to_width_pct,
                max_short_delta=max_short_delta,
                min_underlying_price=min_underlying_price,
                min_debit_spread_price=min_debit_spread_price,
                min_whale_appearances=min_whale_appearances,
            )
            for policy, frame in policy_frames.items():
                for _, row in frame.iterrows():
                    pnl = _safe_float(row.get("quote_replay_pnl"))
                    if not math.isfinite(pnl):
                        continue
                    outcome_rows.append(
                        {
                            "policy": policy,
                            "signal_date": signal_date.isoformat(),
                            "horizon_market_days": int(horizon),
                            "ticker": str(row.get("ticker", "") or "").upper().strip(),
                            "direction": str(row.get("direction", "") or "").strip().lower(),
                            "strategy": str(row.get("strategy", "") or "").strip(),
                            "trade_setup": _trade_setup_text(row),
                            "swing_score": _safe_float(row.get("swing_score")),
                            "edge_pct": _safe_float(row.get("edge_pct")),
                            "backtest_signals": _safe_int(row.get("backtest_signals")),
                            "backtest_verdict": _verdict_text(row.get("backtest_verdict", "")),
                            "price_direction": str(row.get("price_direction", "") or "").strip().lower(),
                            "price_trend": _safe_float(row.get("price_trend")),
                            "flow_direction": str(row.get("flow_direction", "") or "").strip().lower(),
                            "flow_persistence": _safe_float(row.get("flow_persistence")),
                            "oi_direction": str(row.get("oi_direction", "") or "").strip().lower(),
                            "oi_momentum": _safe_float(row.get("oi_momentum")),
                            "dp_direction": str(row.get("dp_direction", "") or "").strip().lower(),
                            "dp_confirmation": _safe_float(row.get("dp_confirmation")),
                            "sector": str(row.get("sector", "") or "").strip(),
                            "whale_appearances": _safe_int(row.get("whale_appearances")),
                            "days_observed": _safe_int(row.get("days_observed")),
                            "latest_close": _safe_float(row.get("latest_close")),
                            "latest_iv_rank": _safe_float(row.get("latest_iv_rank")),
                            "iv_level": str(row.get("iv_level", "") or "").strip(),
                            "cost_type": str(row.get("cost_type", "") or "").strip().lower(),
                            "track": str(row.get("track", "") or "").strip(),
                            "variant_tag": str(row.get("variant_tag", "") or "").strip() or "base",
                            "target_dte": _safe_int(row.get("target_dte")),
                            "base_gate_reasons": str(row.get("base_gate_reasons", "") or "").strip(),
                            "quality_reject_reasons": str(row.get("quality_reject_reasons", "") or "").strip(),
                            "entry_net": _safe_float(row.get("quote_replay_entry_net")),
                            "exit_net": _safe_float(row.get("quote_replay_exit_net")),
                            "pnl": pnl,
                            "return_on_risk": _safe_float(row.get("quote_replay_return_on_risk")),
                        }
                    )
    outcomes = pd.DataFrame(outcome_rows, columns=RESEARCH_OUTCOME_COLUMNS)
    return outcomes


def run_research_confidence_audit(
    *,
    root: Path,
    out_dir: Path,
    as_of: dt.date,
    lookback: int,
    samples: int,
    horizons: List[int],
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
) -> pd.DataFrame:
    outcomes = collect_research_confidence_outcomes(
        root=root,
        out_dir=out_dir,
        as_of=as_of,
        lookback=lookback,
        samples=samples,
        horizons=horizons,
        min_edge=min_edge,
        min_signals=min_signals,
        min_workup_signals=min_workup_signals,
        min_swing_score=min_swing_score,
        allow_low_sample=allow_low_sample,
        allow_earnings_risk=allow_earnings_risk,
        allow_volatile_ic=allow_volatile_ic,
        allow_flow_conflict=allow_flow_conflict,
        max_bid_ask_to_price_pct=max_bid_ask_to_price_pct,
        max_bid_ask_to_width_pct=max_bid_ask_to_width_pct,
        max_short_delta=max_short_delta,
        min_underlying_price=min_underlying_price,
        min_debit_spread_price=min_debit_spread_price,
        min_whale_appearances=min_whale_appearances,
    )
    return _research_summary_from_outcomes(outcomes)


def _growth_sector(value: Any) -> bool:
    return str(value or "").strip() in {
        "Technology",
        "Communication Services",
        "Consumer Cyclical",
    }


def _strategy_family_labels(row: pd.Series) -> List[str]:
    direction = str(row.get("direction", "") or "").strip().lower()
    strategy = str(row.get("strategy", "") or "").strip()
    sector = str(row.get("sector", "") or "").strip()
    price_direction = str(row.get("price_direction", "") or "").strip().lower()
    flow_direction = str(row.get("flow_direction", "") or "").strip().lower()
    oi_direction = str(row.get("oi_direction", "") or "").strip().lower()
    dp_direction = str(row.get("dp_direction", "") or "").strip().lower()
    score = _safe_float(row.get("swing_score"))
    price = _safe_float(row.get("price_trend"))
    flow = _safe_float(row.get("flow_persistence"))
    oi = _safe_float(row.get("oi_momentum"))
    dp = _safe_float(row.get("dp_confirmation"))
    whales = _safe_int(row.get("whale_appearances"))

    labels: List[str] = []
    if (
        direction == "bullish"
        and strategy == "Bull Call Debit"
        and sector in {"Energy", "Basic Materials"}
        and price_direction == "bullish"
        and flow_direction == "bullish"
        and score >= 60
        and price >= 55
        and flow >= 55
    ):
        labels.append("bull_energy_materials_debit")

    if (
        direction == "bearish"
        and strategy == "Bear Put Debit"
        and price_direction == "bearish"
        and flow_direction == "bearish"
        and score >= 60
        and price >= 55
        and flow >= 55
    ):
        labels.append("bear_momentum_put_debit")

    if (
        direction == "bullish"
        and strategy == "Bull Call Debit"
        and price_direction == "bullish"
        and flow_direction == "bullish"
        and score >= 65
        and price >= 65
        and flow >= 65
        and whales >= 8
    ):
        labels.append("bull_quality_momentum_debit")

    if (
        direction == "bullish"
        and strategy == "Bull Call Debit"
        and price_direction == "bullish"
        and flow_direction == "bullish"
        and oi_direction == "bullish"
        and dp_direction == "accumulation"
        and score >= 65
        and oi >= 55
        and dp >= 60
    ):
        labels.append("bull_oi_dp_confirmed_debit")

    if (
        direction == "bullish"
        and strategy == "Bull Put Credit"
        and flow_direction == "bullish"
        and oi_direction == "bullish"
        and score >= 55
    ):
        labels.append("bull_credit_put_support")

    if (
        direction == "bullish"
        and strategy == "Bull Call Debit"
        and _growth_sector(sector)
        and price_direction == "bullish"
        and score >= 60
    ):
        labels.append("growth_bull_call_debit")

    return labels


def _profit_factor(pnl: pd.Series) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses <= 0:
        return math.inf if gains > 0 else math.nan
    return gains / losses


def _max_drawdown_by_date(df: pd.DataFrame) -> float:
    if df.empty or "signal_date" not in df.columns:
        return math.nan
    work = df.copy()
    work["_pnl"] = pd.to_numeric(work.get("pnl", pd.Series(np.nan, index=work.index)), errors="coerce")
    work = work[work["_pnl"].notna()].copy()
    if work.empty:
        return math.nan
    by_day = work.groupby("signal_date")["_pnl"].sum().sort_index()
    equity = by_day.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min())


def _family_period_summary(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    pnl = pd.to_numeric(df.get("pnl", pd.Series(np.nan, index=df.index)), errors="coerce")
    completed = df[pnl.notna()].copy()
    pnl = pnl[pnl.notna()]
    if completed.empty:
        return {
            f"{prefix}_outcomes": 0,
            f"{prefix}_unique_setups": 0,
            f"{prefix}_dates": 0,
            f"{prefix}_hit_rate": math.nan,
            f"{prefix}_avg_pnl": math.nan,
            f"{prefix}_profit_factor": math.nan,
            f"{prefix}_max_drawdown": math.nan,
        }
    return {
        f"{prefix}_outcomes": int(len(completed)),
        f"{prefix}_unique_setups": _research_unique_setup_count(completed),
        f"{prefix}_dates": int(completed.get("signal_date", pd.Series(dtype=str)).nunique()),
        f"{prefix}_hit_rate": float(pnl.gt(0).mean()),
        f"{prefix}_avg_pnl": float(pnl.mean()),
        f"{prefix}_profit_factor": _profit_factor(pnl),
        f"{prefix}_max_drawdown": _max_drawdown_by_date(completed),
    }


def _strategy_family_verdict(row: Dict[str, Any]) -> str:
    train_setups = _safe_int(row.get("train_unique_setups"))
    validation_setups = _safe_int(row.get("validation_unique_setups"))
    validation_dates = _safe_int(row.get("validation_dates"))
    train_avg = _safe_float(row.get("train_avg_pnl"))
    validation_avg = _safe_float(row.get("validation_avg_pnl"))
    train_pf = _safe_float(row.get("train_profit_factor"))
    validation_pf = _safe_float(row.get("validation_profit_factor"))
    validation_hit = _safe_float(row.get("validation_hit_rate"))
    train_pf_ok = (math.isfinite(train_pf) and train_pf >= 1.05) or train_pf == math.inf
    validation_pf_ok = (
        (math.isfinite(validation_pf) and validation_pf >= FAMILY_AUDIT_MIN_PROFIT_FACTOR)
        or validation_pf == math.inf
    )

    if train_setups <= 0 and validation_setups <= 0:
        return "empty"
    if (
        train_setups < FAMILY_AUDIT_MIN_TRAIN_SETUPS
        or validation_setups < FAMILY_AUDIT_MIN_VALIDATION_SETUPS
        or validation_dates < FAMILY_AUDIT_MIN_VALIDATION_DATES
    ):
        if (
            math.isfinite(train_avg)
            and train_avg > 0
            and math.isfinite(validation_avg)
            and validation_avg > 0
        ):
            return "emerging_low_sample"
        return "low_sample"
    if (
        math.isfinite(train_avg)
        and train_avg > 0
        and math.isfinite(validation_avg)
        and validation_avg > 0
        and train_pf_ok
        and validation_pf_ok
        and math.isfinite(validation_hit)
        and validation_hit >= 0.50
    ):
        return "promotable"
    if math.isfinite(train_avg) and train_avg < 0 and math.isfinite(validation_avg) and validation_avg < 0:
        return "negative"
    if math.isfinite(validation_avg) and validation_avg < 0:
        return "validation_negative"
    if math.isfinite(validation_pf) and validation_pf < 1.0:
        return "validation_weak"
    return "mixed"


def _strategy_family_audit_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=STRATEGY_FAMILY_AUDIT_COLUMNS)
    df = outcomes.copy()
    if "policy" in df.columns:
        df = df[df["policy"].fillna("").astype(str).eq("entry_available_score_gate")].copy()
    if df.empty:
        return pd.DataFrame(columns=STRATEGY_FAMILY_AUDIT_COLUMNS)
    for col in (
        "pnl",
        "swing_score",
        "price_trend",
        "flow_persistence",
        "oi_momentum",
        "dp_confirmation",
        "whale_appearances",
    ):
        df[col] = pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")
    df["_families"] = df.apply(_strategy_family_labels, axis=1)
    df = df[df["_families"].map(bool)].copy()
    if df.empty:
        return pd.DataFrame(columns=STRATEGY_FAMILY_AUDIT_COLUMNS)

    exploded = df.explode("_families").rename(columns={"_families": "family"})
    dedupe_cols = [c for c in ["family", "horizon_market_days", "signal_date", "ticker", "trade_setup"] if c in exploded.columns]
    if dedupe_cols:
        exploded = exploded.drop_duplicates(subset=dedupe_cols, keep="first")

    rows: List[Dict[str, Any]] = []
    for (family, horizon), group in exploded.groupby(["family", "horizon_market_days"], dropna=False):
        parsed_horizon = _safe_int(horizon)
        if parsed_horizon <= 0:
            continue
        dates = sorted(str(d) for d in group.get("signal_date", pd.Series(dtype=str)).dropna().unique())
        if not dates:
            continue
        cut = max(1, int(math.floor(len(dates) * 0.70)))
        if len(dates) > 1:
            cut = min(cut, len(dates) - 1)
        train_dates = set(dates[:cut])
        validation_dates = set(dates[cut:])
        train = group[group["signal_date"].astype(str).isin(train_dates)].copy()
        validation = group[group["signal_date"].astype(str).isin(validation_dates)].copy()
        overall = _family_period_summary(group, "overall")
        row: Dict[str, Any] = {
            "family": str(family),
            "horizon_market_days": int(parsed_horizon),
            "description": STRATEGY_FAMILY_DESCRIPTIONS.get(str(family), ""),
            **overall,
            **_family_period_summary(train, "train"),
            **_family_period_summary(validation, "validation"),
            "worst_pnl": float(pd.to_numeric(group.get("pnl", pd.Series(np.nan, index=group.index)), errors="coerce").min()),
        }
        row["verdict"] = _strategy_family_verdict(row)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=STRATEGY_FAMILY_AUDIT_COLUMNS)
    out = pd.DataFrame(rows, columns=STRATEGY_FAMILY_AUDIT_COLUMNS)
    verdict_rank = {
        "promotable": 0,
        "emerging_low_sample": 1,
        "mixed": 2,
        "validation_weak": 3,
        "validation_negative": 4,
        "negative": 5,
        "low_sample": 6,
        "empty": 7,
    }
    out["_rank"] = out["verdict"].map(verdict_rank).fillna(99)
    out["_avg"] = pd.to_numeric(out["validation_avg_pnl"], errors="coerce").fillna(-999999)
    out = out.sort_values(["_rank", "_avg"], ascending=[True, False])
    return out.drop(columns=["_rank", "_avg"], errors="ignore").reset_index(drop=True)


def _family_summary_text(row: pd.Series) -> str:
    verdict = str(row.get("verdict", "") or "").strip()
    horizon = _safe_int(row.get("horizon_market_days"))
    val_avg = _fmt_money(row.get("validation_avg_pnl"))
    val_pf = _safe_float(row.get("validation_profit_factor"))
    val_hit = _safe_float(row.get("validation_hit_rate"))
    val_setups = _safe_int(row.get("validation_unique_setups"))
    parts = [f"{verdict}"]
    if horizon > 0:
        parts.append(f"{horizon}d")
    parts.append(f"validation {val_setups} setups")
    if math.isfinite(val_hit):
        parts.append(f"hit {val_hit:.0%}")
    parts.append(f"avg {val_avg}")
    if math.isfinite(val_pf):
        parts.append(f"PF {val_pf:.2f}")
    return ", ".join(parts)


def _best_family_for_candidate(row: pd.Series, family_audit: pd.DataFrame) -> Optional[pd.Series]:
    labels = _strategy_family_labels(row)
    if not labels or family_audit.empty:
        return None
    subset = family_audit[family_audit["family"].astype(str).isin(labels)].copy()
    if subset.empty:
        return None
    verdict_rank = {
        "promotable": 0,
        "emerging_low_sample": 1,
        "mixed": 2,
        "validation_weak": 3,
        "validation_negative": 4,
        "negative": 5,
        "low_sample": 6,
        "empty": 7,
    }
    subset["_rank"] = subset["verdict"].map(verdict_rank).fillna(99)
    subset["_val_avg"] = pd.to_numeric(subset["validation_avg_pnl"], errors="coerce").fillna(-999999)
    subset = subset.sort_values(["_rank", "_val_avg"], ascending=[True, False])
    return subset.iloc[0]


def annotate_strategy_family_gate(candidates: pd.DataFrame, family_audit: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    defaults = {
        "strategy_family": "",
        "strategy_family_horizon": 0,
        "strategy_family_verdict": "no_match",
        "strategy_family_gate_pass": False,
        "strategy_family_summary": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if family_audit.empty:
        out["strategy_family_verdict"] = "no_audit"
        out["strategy_family_summary"] = "no completed strategy-family audit"
        return out

    for idx, row in out.iterrows():
        best = _best_family_for_candidate(row, family_audit)
        if best is None:
            labels = _strategy_family_labels(row)
            out.at[idx, "strategy_family"] = ",".join(labels)
            out.at[idx, "strategy_family_verdict"] = "no_promoted_match" if labels else "no_match"
            out.at[idx, "strategy_family_summary"] = (
                "candidate does not match a predeclared audited family"
                if not labels
                else "matched family has no completed audit row"
            )
            continue
        out.at[idx, "strategy_family"] = str(best.get("family", "") or "")
        out.at[idx, "strategy_family_horizon"] = _safe_int(best.get("horizon_market_days"))
        verdict = str(best.get("verdict", "") or "").strip()
        out.at[idx, "strategy_family_verdict"] = verdict
        out.at[idx, "strategy_family_gate_pass"] = verdict == "promotable"
        out.at[idx, "strategy_family_summary"] = _family_summary_text(best)
    return out


def _strategy_family_supportive(df: pd.DataFrame) -> bool:
    if df.empty or "verdict" not in df.columns:
        return False
    return bool(df["verdict"].fillna("").astype(str).eq("promotable").any())


def _ticker_playbook_verdict(row: Dict[str, Any]) -> str:
    train_setups = _safe_int(row.get("train_unique_setups"))
    validation_setups = _safe_int(row.get("validation_unique_setups"))
    validation_dates = _safe_int(row.get("validation_dates"))
    train_avg = _safe_float(row.get("train_avg_pnl"))
    validation_avg = _safe_float(row.get("validation_avg_pnl"))
    train_pf = _safe_float(row.get("train_profit_factor"))
    validation_pf = _safe_float(row.get("validation_profit_factor"))
    validation_hit = _safe_float(row.get("validation_hit_rate"))
    train_pf_ok = (math.isfinite(train_pf) and train_pf >= 1.05) or train_pf == math.inf
    validation_pf_ok = (
        (math.isfinite(validation_pf) and validation_pf >= TICKER_PLAYBOOK_MIN_PROFIT_FACTOR)
        or validation_pf == math.inf
    )

    if train_setups <= 0 and validation_setups <= 0:
        return "empty"
    if (
        train_setups < TICKER_PLAYBOOK_MIN_TRAIN_SETUPS
        or validation_setups < TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS
        or validation_dates < TICKER_PLAYBOOK_MIN_VALIDATION_DATES
    ):
        if (
            math.isfinite(train_avg)
            and train_avg > 0
            and math.isfinite(validation_avg)
            and validation_avg > 0
        ):
            return "emerging_low_sample"
        return "low_sample"
    if (
        math.isfinite(train_avg)
        and train_avg > 0
        and math.isfinite(validation_avg)
        and validation_avg > 0
        and train_pf_ok
        and validation_pf_ok
        and math.isfinite(validation_hit)
        and validation_hit >= TICKER_PLAYBOOK_MIN_VALIDATION_HIT
    ):
        return "promotable"
    if math.isfinite(train_avg) and train_avg <= 0 and math.isfinite(validation_avg) and validation_avg > 0:
        return "mixed"
    if math.isfinite(validation_avg) and validation_avg < 0:
        return "validation_negative"
    if math.isfinite(validation_pf) and validation_pf < 1.0:
        return "validation_weak"
    return "mixed"


def _ticker_playbook_audit_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    df = outcomes.copy()
    if "policy" in df.columns:
        df = df[df["policy"].fillna("").astype(str).eq("entry_available_score_gate")].copy()
    if df.empty:
        return pd.DataFrame(columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    for col in ("pnl", "horizon_market_days"):
        df[col] = pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")
    required = ["ticker", "direction", "strategy", "horizon_market_days", "signal_date", "trade_setup"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    df = df[df["pnl"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    dedupe_cols = ["ticker", "direction", "strategy", "horizon_market_days", "signal_date", "trade_setup"]
    df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    rows: List[Dict[str, Any]] = []
    group_cols = ["ticker", "direction", "strategy", "horizon_market_days"]
    for (ticker, direction, strategy, horizon), group in df.groupby(group_cols, dropna=False):
        parsed_horizon = _safe_int(horizon)
        if parsed_horizon <= 0:
            continue
        dates = sorted(str(d) for d in group.get("signal_date", pd.Series(dtype=str)).dropna().unique())
        if len(dates) < 2:
            continue
        cut = max(1, int(math.floor(len(dates) * 0.70)))
        cut = min(cut, len(dates) - 1)
        train_dates = set(dates[:cut])
        validation_dates = set(dates[cut:])
        train = group[group["signal_date"].astype(str).isin(train_dates)].copy()
        validation = group[group["signal_date"].astype(str).isin(validation_dates)].copy()
        overall = _family_period_summary(group, "overall")
        row: Dict[str, Any] = {
            "ticker": str(ticker).strip().upper(),
            "direction": str(direction or "").strip().lower(),
            "strategy": str(strategy or "").strip(),
            "horizon_market_days": int(parsed_horizon),
            "description": "Ticker-specific option replay playbook, split chronologically into train and validation.",
            **overall,
            **_family_period_summary(train, "train"),
            **_family_period_summary(validation, "validation"),
            "worst_pnl": float(pd.to_numeric(group.get("pnl", pd.Series(np.nan, index=group.index)), errors="coerce").min()),
        }
        row["verdict"] = _ticker_playbook_verdict(row)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    out = pd.DataFrame(rows, columns=TICKER_PLAYBOOK_AUDIT_COLUMNS)
    verdict_rank = {
        "promotable": 0,
        "emerging_low_sample": 1,
        "mixed": 2,
        "validation_weak": 3,
        "validation_negative": 4,
        "negative": 5,
        "low_sample": 6,
        "empty": 7,
    }
    out["_rank"] = out["verdict"].map(verdict_rank).fillna(99)
    out["_avg"] = pd.to_numeric(out["validation_avg_pnl"], errors="coerce").fillna(-999999)
    out = out.sort_values(["_rank", "_avg"], ascending=[True, False])
    return out.drop(columns=["_rank", "_avg"], errors="ignore").reset_index(drop=True)


def _best_ticker_playbook_for_candidate(row: pd.Series, playbook_audit: pd.DataFrame) -> Optional[pd.Series]:
    if playbook_audit.empty:
        return None
    ticker = str(row.get("ticker", "") or "").strip().upper()
    direction = str(row.get("direction", "") or "").strip().lower()
    strategy = str(row.get("strategy", "") or "").strip()
    subset = playbook_audit[
        playbook_audit["ticker"].fillna("").astype(str).str.upper().eq(ticker)
        & playbook_audit["direction"].fillna("").astype(str).str.lower().eq(direction)
        & playbook_audit["strategy"].fillna("").astype(str).eq(strategy)
    ].copy()
    if subset.empty:
        return None
    verdict_rank = {
        "promotable": 0,
        "emerging_low_sample": 1,
        "mixed": 2,
        "validation_weak": 3,
        "validation_negative": 4,
        "negative": 5,
        "low_sample": 6,
        "empty": 7,
    }
    subset["_rank"] = subset["verdict"].map(verdict_rank).fillna(99)
    subset["_val_avg"] = pd.to_numeric(subset["validation_avg_pnl"], errors="coerce").fillna(-999999)
    subset = subset.sort_values(["_rank", "_val_avg"], ascending=[True, False])
    return subset.iloc[0]


def annotate_ticker_playbook_gate(candidates: pd.DataFrame, playbook_audit: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    defaults = {
        "ticker_playbook_horizon": 0,
        "ticker_playbook_train_setups": 0,
        "ticker_playbook_validation_setups": 0,
        "ticker_playbook_validation_hit_rate": math.nan,
        "ticker_playbook_validation_avg_pnl": math.nan,
        "ticker_playbook_verdict": "no_match",
        "ticker_playbook_gate_pass": False,
        "ticker_playbook_summary": "",
    }
    for col, value in defaults.items():
        out[col] = value
    if playbook_audit.empty:
        out["ticker_playbook_verdict"] = "no_audit"
        out["ticker_playbook_summary"] = "no completed ticker playbook audit"
        return out

    for idx, row in out.iterrows():
        best = _best_ticker_playbook_for_candidate(row, playbook_audit)
        if best is None:
            out.at[idx, "ticker_playbook_summary"] = "no ticker-specific historical playbook match"
            continue
        verdict = str(best.get("verdict", "") or "").strip()
        out.at[idx, "ticker_playbook_horizon"] = _safe_int(best.get("horizon_market_days"))
        out.at[idx, "ticker_playbook_train_setups"] = _safe_int(best.get("train_unique_setups"))
        out.at[idx, "ticker_playbook_validation_setups"] = _safe_int(best.get("validation_unique_setups"))
        out.at[idx, "ticker_playbook_validation_hit_rate"] = _safe_float(best.get("validation_hit_rate"))
        out.at[idx, "ticker_playbook_validation_avg_pnl"] = _safe_float(best.get("validation_avg_pnl"))
        out.at[idx, "ticker_playbook_verdict"] = verdict
        out.at[idx, "ticker_playbook_gate_pass"] = verdict == "promotable"
        out.at[idx, "ticker_playbook_summary"] = _family_summary_text(best)
    return out


def _ticker_playbook_supportive(df: pd.DataFrame) -> bool:
    if df.empty or "verdict" not in df.columns:
        return False
    return bool(df["verdict"].fillna("").astype(str).eq("promotable").any())


def _rolling_forward_verdict(row: Dict[str, Any]) -> str:
    tests = _safe_int(row.get("forward_tests"))
    dates = _safe_int(row.get("forward_dates"))
    avg = _safe_float(row.get("forward_avg_pnl"))
    pf = _safe_float(row.get("forward_profit_factor"))
    hit = _safe_float(row.get("forward_hit_rate"))
    pf_ok = (math.isfinite(pf) and pf >= ROLLING_PLAYBOOK_MIN_PROFIT_FACTOR) or pf == math.inf
    if tests <= 0:
        return "insufficient_forward"
    if tests < ROLLING_PLAYBOOK_MIN_FORWARD_TESTS or dates < ROLLING_PLAYBOOK_MIN_FORWARD_DATES:
        return "emerging_forward" if math.isfinite(avg) and avg > 0 else "insufficient_forward"
    if (
        math.isfinite(avg)
        and avg > 0
        and pf_ok
        and math.isfinite(hit)
        and hit >= ROLLING_PLAYBOOK_MIN_HIT_RATE
    ):
        return "supportive"
    return "negative"


def _rolling_ticker_playbook_audit_from_outcomes(
    outcomes: pd.DataFrame,
    playbook_audit: pd.DataFrame,
) -> pd.DataFrame:
    if playbook_audit.empty:
        return pd.DataFrame(columns=ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS)
    rows: List[Dict[str, Any]] = []
    if outcomes.empty:
        for _, playbook in playbook_audit.iterrows():
            row = {
                "ticker": str(playbook.get("ticker", "") or "").strip().upper(),
                "direction": str(playbook.get("direction", "") or "").strip().lower(),
                "strategy": str(playbook.get("strategy", "") or "").strip(),
                "horizon_market_days": _safe_int(playbook.get("horizon_market_days")),
                "forward_tests": 0,
                "forward_dates": 0,
                "forward_hit_rate": math.nan,
                "forward_avg_pnl": math.nan,
                "forward_profit_factor": math.nan,
                "forward_worst_pnl": math.nan,
                "first_forward_date": "",
                "last_forward_date": "",
            }
            row["verdict"] = _rolling_forward_verdict(row)
            rows.append(row)
        return pd.DataFrame(rows, columns=ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS)

    df = outcomes.copy()
    if "policy" in df.columns:
        df = df[df["policy"].fillna("").astype(str).eq("entry_available_score_gate")].copy()
    if df.empty:
        return pd.DataFrame(columns=ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS)
    required = ["ticker", "direction", "strategy", "horizon_market_days", "signal_date", "trade_setup", "pnl"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["horizon_market_days"] = pd.to_numeric(df["horizon_market_days"], errors="coerce")
    df = df[df["pnl"].notna()].drop_duplicates(
        subset=["ticker", "direction", "strategy", "horizon_market_days", "signal_date", "trade_setup"],
        keep="first",
    )

    for _, playbook in playbook_audit.iterrows():
        ticker = str(playbook.get("ticker", "") or "").strip().upper()
        direction = str(playbook.get("direction", "") or "").strip().lower()
        strategy = str(playbook.get("strategy", "") or "").strip()
        horizon = _safe_int(playbook.get("horizon_market_days"))
        group = df[
            df["ticker"].fillna("").astype(str).str.upper().eq(ticker)
            & df["direction"].fillna("").astype(str).str.lower().eq(direction)
            & df["strategy"].fillna("").astype(str).eq(strategy)
            & pd.to_numeric(df["horizon_market_days"], errors="coerce").fillna(0).astype(int).eq(horizon)
        ].copy()
        forward_parts: List[pd.DataFrame] = []
        if not group.empty:
            dates = sorted(str(d) for d in group["signal_date"].dropna().unique())
            for signal_date in dates:
                prior = group[group["signal_date"].astype(str) < signal_date].copy()
                current = group[group["signal_date"].astype(str).eq(signal_date)].copy()
                if prior.empty or current.empty:
                    continue
                prior_audit = _ticker_playbook_audit_from_outcomes(prior.assign(policy="entry_available_score_gate"))
                if prior_audit.empty:
                    continue
                if str(prior_audit.iloc[0].get("verdict", "") or "") == "promotable":
                    forward_parts.append(current)
        if forward_parts:
            forward = pd.concat(forward_parts, ignore_index=True)
            pnl = pd.to_numeric(forward["pnl"], errors="coerce").dropna()
            dates = sorted(str(d) for d in forward["signal_date"].dropna().unique())
            row = {
                "ticker": ticker,
                "direction": direction,
                "strategy": strategy,
                "horizon_market_days": horizon,
                "forward_tests": int(len(forward)),
                "forward_dates": int(forward["signal_date"].nunique()),
                "forward_hit_rate": float(pnl.gt(0).mean()) if len(pnl) else math.nan,
                "forward_avg_pnl": float(pnl.mean()) if len(pnl) else math.nan,
                "forward_profit_factor": _profit_factor(pnl) if len(pnl) else math.nan,
                "forward_worst_pnl": float(pnl.min()) if len(pnl) else math.nan,
                "first_forward_date": dates[0] if dates else "",
                "last_forward_date": dates[-1] if dates else "",
            }
        else:
            row = {
                "ticker": ticker,
                "direction": direction,
                "strategy": strategy,
                "horizon_market_days": horizon,
                "forward_tests": 0,
                "forward_dates": 0,
                "forward_hit_rate": math.nan,
                "forward_avg_pnl": math.nan,
                "forward_profit_factor": math.nan,
                "forward_worst_pnl": math.nan,
                "first_forward_date": "",
                "last_forward_date": "",
            }
        row["verdict"] = _rolling_forward_verdict(row)
        rows.append(row)

    out = pd.DataFrame(rows, columns=ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS)
    verdict_rank = {"supportive": 0, "emerging_forward": 1, "insufficient_forward": 2, "negative": 3}
    out["_rank"] = out["verdict"].map(verdict_rank).fillna(9)
    out["_avg"] = pd.to_numeric(out["forward_avg_pnl"], errors="coerce").fillna(-999999)
    return out.sort_values(["_rank", "_avg"], ascending=[True, False]).drop(
        columns=["_rank", "_avg"],
        errors="ignore",
    ).reset_index(drop=True)


def _rolling_summary_text(row: pd.Series) -> str:
    verdict = str(row.get("verdict", "") or "").strip()
    tests = _safe_int(row.get("forward_tests"))
    dates = _safe_int(row.get("forward_dates"))
    hit = _safe_float(row.get("forward_hit_rate"))
    avg = _fmt_money(row.get("forward_avg_pnl"))
    pf = _safe_float(row.get("forward_profit_factor"))
    parts = [verdict, f"{tests} forward tests", f"{dates} dates"]
    if math.isfinite(hit):
        parts.append(f"hit {hit:.0%}")
    parts.append(f"avg {avg}")
    if math.isfinite(pf):
        parts.append(f"PF {pf:.2f}")
    return ", ".join(parts)


def annotate_rolling_playbook_gate(candidates: pd.DataFrame, rolling_audit: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["rolling_playbook_verdict"] = "no_audit"
    out["rolling_playbook_gate_pass"] = True
    out["rolling_playbook_summary"] = "no rolling ticker playbook audit"
    if rolling_audit.empty:
        return out
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        direction = str(row.get("direction", "") or "").strip().lower()
        strategy = str(row.get("strategy", "") or "").strip()
        horizon = _safe_int(row.get("ticker_playbook_horizon"))
        subset = rolling_audit[
            rolling_audit["ticker"].fillna("").astype(str).str.upper().eq(ticker)
            & rolling_audit["direction"].fillna("").astype(str).str.lower().eq(direction)
            & rolling_audit["strategy"].fillna("").astype(str).eq(strategy)
        ].copy()
        if horizon > 0 and not subset.empty:
            exact = subset[pd.to_numeric(subset["horizon_market_days"], errors="coerce").fillna(0).astype(int).eq(horizon)]
            if not exact.empty:
                subset = exact
        if subset.empty:
            out.at[idx, "rolling_playbook_verdict"] = "no_match"
            out.at[idx, "rolling_playbook_summary"] = "no rolling ticker playbook match"
            continue
        verdict_rank = {"supportive": 0, "emerging_forward": 1, "insufficient_forward": 2, "negative": 3}
        subset["_rank"] = subset["verdict"].map(verdict_rank).fillna(9)
        subset["_avg"] = pd.to_numeric(subset["forward_avg_pnl"], errors="coerce").fillna(-999999)
        best = subset.sort_values(["_rank", "_avg"], ascending=[True, False]).iloc[0]
        verdict = str(best.get("verdict", "") or "").strip()
        out.at[idx, "rolling_playbook_verdict"] = verdict
        out.at[idx, "rolling_playbook_gate_pass"] = verdict != "negative"
        out.at[idx, "rolling_playbook_summary"] = _rolling_summary_text(best)
    return out


def _ticker_playbook_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no completed ticker playbook audit rows were available_"]
    lines = [
        "This checks ticker-specific setups so a profitable ticker playbook is not buried inside a broad losing sector/family bucket. It is still split into earlier training dates and later validation dates.",
        "",
        _render_table(
            [
                "ticker",
                "direction",
                "strategy",
                "horizon",
                "train setups",
                "train avg",
                "train PF",
                "validation setups",
                "validation hit",
                "validation avg",
                "validation PF",
                "worst",
                "verdict",
            ],
            [
                [
                    str(row.get("ticker", "")),
                    str(row.get("direction", "")),
                    str(row.get("strategy", "")),
                    str(_safe_int(row.get("horizon_market_days"))),
                    str(_safe_int(row.get("train_unique_setups"))),
                    _fmt_money(row.get("train_avg_pnl")),
                    _fmt_num(row.get("train_profit_factor"), 2),
                    str(_safe_int(row.get("validation_unique_setups"))),
                    f"{_safe_float(row.get('validation_hit_rate')):.0%}"
                    if math.isfinite(_safe_float(row.get("validation_hit_rate")))
                    else "-",
                    _fmt_money(row.get("validation_avg_pnl")),
                    _fmt_num(row.get("validation_profit_factor"), 2),
                    _fmt_money(row.get("worst_pnl")),
                    str(row.get("verdict", "")),
                ]
                for _, row in df.head(16).iterrows()
            ],
        ),
    ]
    if not _ticker_playbook_supportive(df):
        lines.extend(
            [
                "",
                "**Confidence impact:** no ticker playbook is promotable yet, so order-ready trades stay blocked.",
            ]
        )
    return lines


def _rolling_ticker_playbook_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no rolling ticker-playbook forward validation rows were available_"]
    lines = [
        "This replays ticker playbooks chronologically: a playbook must become promotable using only earlier dates before the next date is scored as forward validation.",
        "",
        _render_table(
            [
                "ticker",
                "direction",
                "strategy",
                "horizon",
                "forward tests",
                "dates",
                "hit",
                "avg",
                "PF",
                "worst",
                "window",
                "verdict",
            ],
            [
                [
                    str(row.get("ticker", "")),
                    str(row.get("direction", "")),
                    str(row.get("strategy", "")),
                    str(_safe_int(row.get("horizon_market_days"))),
                    str(_safe_int(row.get("forward_tests"))),
                    str(_safe_int(row.get("forward_dates"))),
                    f"{_safe_float(row.get('forward_hit_rate')):.0%}"
                    if math.isfinite(_safe_float(row.get("forward_hit_rate")))
                    else "-",
                    _fmt_money(row.get("forward_avg_pnl")),
                    _fmt_num(row.get("forward_profit_factor"), 2),
                    _fmt_money(row.get("forward_worst_pnl")),
                    (
                        f"{row.get('first_forward_date', '')} to {row.get('last_forward_date', '')}"
                        if str(row.get("first_forward_date", "") or "").strip()
                        else "-"
                    ),
                    str(row.get("verdict", "")),
                ]
                for _, row in df.head(16).iterrows()
            ],
        ),
    ]
    verdicts = df.get("verdict", pd.Series(dtype=str)).fillna("").astype(str)
    if verdicts.eq("negative").any():
        lines.extend(
            [
                "",
                "**Confidence impact:** negative rolling playbooks block matching actionable trades.",
            ]
        )
    return lines


def _strategy_family_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no completed strategy-family audit rows were available_"]
    lines = [
        "This is the trade-generation layer: only predeclared setup families are evaluated, each split into earlier training dates and later validation dates. Only `promotable` families can unlock Actionable Now.",
        "",
        _render_table(
            [
                "family",
                "horizon",
                "train setups",
                "train avg",
                "train PF",
                "validation setups",
                "validation hit",
                "validation avg",
                "validation PF",
                "drawdown",
                "verdict",
            ],
            [
                [
                    str(row.get("family", "")),
                    str(_safe_int(row.get("horizon_market_days"))),
                    str(_safe_int(row.get("train_unique_setups"))),
                    _fmt_money(row.get("train_avg_pnl")),
                    _fmt_num(row.get("train_profit_factor"), 2),
                    str(_safe_int(row.get("validation_unique_setups"))),
                    f"{_safe_float(row.get('validation_hit_rate')):.0%}"
                    if math.isfinite(_safe_float(row.get("validation_hit_rate")))
                    else "-",
                    _fmt_money(row.get("validation_avg_pnl")),
                    _fmt_num(row.get("validation_profit_factor"), 2),
                    _fmt_money(row.get("validation_max_drawdown")),
                    str(row.get("verdict", "")),
                ]
                for _, row in df.head(16).iterrows()
            ],
        ),
    ]
    if not _strategy_family_supportive(df):
        lines.extend(
            [
                "",
                "**Confidence impact:** no predeclared family is promotable yet. Keep order-ready trades blocked; use emerging families only as research targets.",
            ]
        )
    return lines


def _research_audit_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no completed historical research-audit outcomes were available_"]
    lines = [
        "This is a fixed-bucket audit, not a parameter search. A setup tested at several horizons is still one distinct setup; supportive buckets require enough unique setups, not just repeated horizons.",
        "",
        _render_table(
            ["policy", "outcomes", "setups", "horiz/setup", "hit", "avg pnl", "avg R/R", "worst", "verdict"],
            [
                [
                    str(row.get("policy", "")),
                    str(_safe_int(row.get("outcomes"))),
                    str(_safe_int(row.get("unique_setups"))),
                    _fmt_num(row.get("avg_horizons_per_setup"), 1),
                    f"{_safe_float(row.get('hit_rate')):.0%}" if math.isfinite(_safe_float(row.get("hit_rate"))) else "-",
                    _fmt_money(row.get("avg_pnl")),
                    f"{_safe_float(row.get('avg_return_on_risk')):.1%}"
                    if math.isfinite(_safe_float(row.get("avg_return_on_risk")))
                    else "-",
                    _fmt_money(row.get("worst_pnl")),
                    str(row.get("verdict", "")),
                ]
                for _, row in df.iterrows()
            ],
        ),
    ]
    if not (df.get("verdict", pd.Series(dtype=str)).astype(str).eq("supportive")).any():
        lines.extend(
            [
                "",
                "**Confidence impact:** no fixed audit bucket is supportive yet, so this pipeline should continue to block Actionable Now trades.",
            ]
        )
    pass_bucket = df[df.get("policy", pd.Series(dtype=str)).astype(str).eq("backtest_pass_sample_gate")]
    if not pass_bucket.empty and str(pass_bucket.iloc[0].get("verdict", "")).lower() == "negative":
        lines.extend(
            [
                "",
                "**Risk warning:** the likelihood backtest PASS bucket lost money in option-quote replay, so PASS is only a pattern filter here, not profitability validation.",
            ]
        )
    return lines


def _research_horizon_report_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_no per-horizon research outcomes were available_"]
    lines = [
        "This split prevents one setup from looking stronger just because it was scored at 5, 10, and 20 market-day exits.",
        "",
        _render_table(
            ["policy", "horizon", "outcomes", "setups", "hit", "avg pnl", "avg R/R", "worst", "verdict"],
            [
                [
                    str(row.get("policy", "")),
                    str(_safe_int(row.get("horizon_market_days"))),
                    str(_safe_int(row.get("outcomes"))),
                    str(_safe_int(row.get("unique_setups"))),
                    f"{_safe_float(row.get('hit_rate')):.0%}" if math.isfinite(_safe_float(row.get("hit_rate"))) else "-",
                    _fmt_money(row.get("avg_pnl")),
                    f"{_safe_float(row.get('avg_return_on_risk')):.1%}"
                    if math.isfinite(_safe_float(row.get("avg_return_on_risk")))
                    else "-",
                    _fmt_money(row.get("worst_pnl")),
                    str(row.get("verdict", "")),
                ]
                for _, row in df.iterrows()
            ],
        ),
    ]
    if not (df.get("verdict", pd.Series(dtype=str)).astype(str).eq("supportive")).any():
        lines.extend(
            [
                "",
                "**Confidence impact:** no horizon-specific bucket is supportive, so Actionable Now should stay blocked.",
            ]
        )
    return lines


def _best_trade_status(ticker_rows: pd.DataFrame) -> str:
    if ticker_rows.empty:
        return "Candidate only"
    row, support = _best_candidate_trade_row(ticker_rows)
    if row is None:
        return "Backtest rejected: no supported structure yet"
    if support == "actionable":
        return f"Actionable structure found: {_entry_text(row)} exp {row.get('target_expiry', '-')}"

    if support == "risk_blocked":
        reason = str(row.get("quality_reject_reasons", "") or "risk gate").strip()
        return f"Valid pattern, trade blocked: {_clip_text(reason, 120)}"

    if support == "quote_blocked":
        verdict = str(row.get("quote_replay_verdict", "") or "UNAVAILABLE").strip().upper()
        reason = str(row.get("quote_replay_reason", "") or "").strip()
        return f"Backtest supported, but quote replay blocked: {verdict}" + (
            f" ({_clip_text(reason, 100)})" if reason else ""
        )

    if support == "live_blocked":
        note = str(row.get("live_validation_note", "") or "").strip()
        return "Backtest supported, but Schwab live validation failed" + (
            f": {_clip_text(note, 120)}" if note else ""
        )
    return "Candidate only: build or wait for better structure"


def _candidate_work_item(status: str) -> str:
    lower = status.lower()
    if "actionable structure found" in lower:
        return "Review the actionable structure; do not replace the candidate with a weaker trade."
    if "earnings" in lower:
        return "Work after earnings or rebuild with a truly pre-earnings expiry."
    if "wide market" in lower or "liquidity" in lower:
        return "Wait for tighter markets or rebuild around more liquid strikes/expiry."
    if "lotto underlying" in lower or "lotto debit" in lower:
        return "Reject for now: not a professional-quality setup at this price/structure."
    if "flow conflict" in lower:
        return "Reject for now: trade direction conflicts with flow."
    if "thin institutional confirmation" in lower:
        return "Watch only: needs broader institutional confirmation before trade construction."
    if "delta" in lower:
        return "Move short legs farther OTM or switch to a directional debit spread."
    if "quote replay" in lower:
        return "Do not trade until both legs can be replayed from daily option snapshots or a fresher structure is built."
    if "live" in lower:
        return "Reprice with Schwab; current chain/entry did not validate."
    return "Research thesis and construct a fresh spread; not a trade yet."


def build_candidate_shortlist(
    candidates: pd.DataFrame,
    *,
    top_n: int = DEFAULT_CANDIDATE_TOP,
    min_score: float = DEFAULT_CANDIDATE_MIN_SCORE,
    min_confirmations: int = DEFAULT_CANDIDATE_MIN_CONFIRMATIONS,
    max_conflicts: int = DEFAULT_CANDIDATE_MAX_CONFLICTS,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()
    if "variant_tag" in df.columns:
        base = df[df["variant_tag"].fillna("base").astype(str).eq("base")].copy()
    else:
        base = df.copy()
    if base.empty:
        return base

    rows: List[Dict[str, Any]] = []
    for _, row in base.iterrows():
        direction = str(row.get("direction", "") or "").strip().lower()
        score = _safe_float(row.get("swing_score"))
        days = _safe_int(row.get("days_observed"))
        confirmations, conflicts = candidate_evidence(row)
        if direction not in {"bullish", "bearish"}:
            continue
        if not math.isfinite(score) or score < min_score:
            continue
        if days <= 0:
            continue
        if len(confirmations) < int(min_confirmations):
            continue
        if len(conflicts) > int(max_conflicts):
            continue

        ticker = str(row.get("ticker", "") or "").strip().upper()
        ticker_rows = df[df.get("ticker", pd.Series("", index=df.index)).astype(str).str.upper().eq(ticker)]
        best_trade, support = _best_candidate_trade_row(ticker_rows)
        if best_trade is None:
            continue
        status = _best_trade_status(ticker_rows)
        out = best_trade.to_dict()
        # Evidence should come from the base pattern row, even when the best
        # tested structure is a repair variant.
        for evidence_col in (
            "direction",
            "swing_score",
            "flow_persistence",
            "oi_momentum",
            "iv_regime",
            "price_trend",
            "whale_consensus",
            "dp_confirmation",
            "confidence_tier",
            "days_observed",
            "latest_close",
            "latest_iv_rank",
            "iv_level",
            "iv_regime_label",
            "price_direction",
            "flow_direction",
            "oi_direction",
            "dp_direction",
            "whale_appearances",
            "sector",
            "thesis",
        ):
            if evidence_col in row:
                out[evidence_col] = row.get(evidence_col)
        out["candidate_confirmations"] = "; ".join(confirmations)
        out["candidate_conflicts"] = "; ".join(conflicts) if conflicts else "none"
        out["candidate_confirmation_count"] = len(confirmations)
        out["candidate_conflict_count"] = len(conflicts)
        out["candidate_support"] = support
        out["candidate_status"] = status
        next_step = _candidate_work_item(status)
        if any(str(conflict).startswith("price divergence:") for conflict in conflicts):
            next_step = (
                "Treat as divergence: confirm reversal thesis and avoid entry unless price starts confirming."
            )
        out["candidate_next_step"] = next_step
        out["_candidate_sort"] = score + len(confirmations) * 2 - len(conflicts) * 4
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        ["_candidate_sort", "swing_score", "candidate_confirmation_count"],
        ascending=[False, False, False],
    )
    out_df = out_df.drop_duplicates("ticker").head(max(1, int(top_n)))
    return out_df.drop(columns=["_candidate_sort"], errors="ignore").reset_index(drop=True)


def _fmt_num(value: Any, digits: int = 1) -> str:
    parsed = _safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}f}"


def _fmt_money(value: Any) -> str:
    parsed = _safe_float(value)
    if not math.isfinite(parsed):
        return "-"
    return f"${parsed:,.2f}"


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _entry_text(row: pd.Series) -> str:
    live_setup = _clean_cell_text(row.get("live_strike_setup", ""))
    if _truthy(row.get("live_validated")) and live_setup:
        return live_setup
    return _clean_cell_text(row.get("strike_setup", "")) or "-"


def _trade_setup_text(row: pd.Series) -> str:
    parts = []
    strategy = _clean_cell_text(row.get("strategy", ""))
    entry = _entry_text(row)
    expiry = _clean_cell_text(row.get("target_expiry", ""))
    if strategy:
        parts.append(strategy)
    if entry and entry != "-":
        parts.append(entry)
    if expiry:
        parts.append(f"exp {expiry}")
    return " | ".join(parts) or "-"


def _variant_text(row: pd.Series) -> str:
    tag = str(row.get("variant_tag", "") or "base").strip()
    source = str(row.get("repair_source", "") or "").strip()
    if not tag or tag == "base":
        return "base"
    return f"{tag}: {source}" if source else tag


def _verdict_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    return text


def _gate_text(row: pd.Series, *, schwab_enabled: bool, backtest_enabled: bool) -> str:
    parts = []
    if backtest_enabled:
        verdict = _verdict_text(row.get("backtest_verdict", ""))
        edge = _fmt_num(row.get("edge_pct"), 1)
        signals = _safe_int(row.get("backtest_signals"))
        parts.append(f"backtest {verdict}, edge {edge}%, signals {signals}")
        if _ticker_playbook_support_passes(row, min_edge=0.0) and verdict != "PASS":
            summary = str(row.get("ticker_playbook_summary", "") or "").strip()
            parts.append(f"ticker playbook support{': ' + summary if summary else ''}")
    else:
        parts.append("backtest skipped")
    if schwab_enabled:
        live = row.get("live_validated")
        if _truthy(live):
            parts.append(f"Schwab live ok at {_fmt_money(row.get('live_spread_cost'))}")
        else:
            note = str(row.get("live_validation_note", "") or "").strip()
            parts.append(f"Schwab live not validated{': ' + note if note else ''}")
    if "quote_replay_verdict" in row.index:
        verdict = str(row.get("quote_replay_verdict", "") or "UNAVAILABLE").strip().upper()
        pnl = _fmt_money(row.get("quote_replay_pnl"))
        exit_date = _clean_cell_text(row.get("quote_replay_exit_date", ""))
        parts.append(f"quote replay {verdict}{' to ' + exit_date if exit_date else ''}, P&L {pnl}")
    return "; ".join(parts)


def _render_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cleaned = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip(" ;,.") + "..."


def _actionable_rows(df: pd.DataFrame, *, schwab_enabled: bool, backtest_enabled: bool) -> List[List[str]]:
    rows: List[List[str]] = []
    for idx, row in df.iterrows():
        rows.append(
            [
                str(idx + 1),
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                str(row.get("strategy", "")),
                _variant_text(row),
                _entry_text(row),
                str(row.get("target_expiry", "")),
                _fmt_num(row.get("swing_score"), 1),
                _fmt_num(row.get("edge_pct"), 1),
                str(_safe_int(row.get("backtest_signals"))),
                _gate_text(row, schwab_enabled=schwab_enabled, backtest_enabled=backtest_enabled),
            ]
        )
    return rows


def _pattern_rows(df: pd.DataFrame, *, limit: int = 15) -> List[List[str]]:
    rows: List[List[str]] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        rows.append(
            [
                str(number),
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                str(row.get("strategy", "")),
                _variant_text(row),
                _fmt_num(row.get("swing_score"), 1),
                _verdict_text(row.get("backtest_verdict", "")),
                _fmt_num(row.get("edge_pct"), 1),
                str(_safe_int(row.get("backtest_signals"))),
                _clip_text(row.get("actionability_reject_reasons", "") or "-", 220),
                _clip_text(row.get("thesis", ""), 220),
            ]
        )
    return rows


def _risk_blocked_rows(df: pd.DataFrame, *, limit: int = 20) -> List[List[str]]:
    rows: List[List[str]] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        rows.append(
            [
                str(number),
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                str(row.get("strategy", "")),
                _variant_text(row),
                _entry_text(row),
                str(row.get("target_expiry", "")),
                _fmt_num(row.get("edge_pct"), 1),
                str(_safe_int(row.get("backtest_signals"))),
                _clip_text(row.get("quality_reject_reasons", "") or "-", 420),
            ]
        )
    return rows


def _watchlist_trigger(row: pd.Series) -> str:
    reasons = str(row.get("actionability_reject_reasons", "") or "").lower()
    quality_reasons = str(row.get("quality_reject_reasons", "") or "").lower()
    all_reasons = f"{reasons}; {quality_reasons}"
    strategy = str(row.get("strategy", "") or "").strip().lower()

    if "backtest fail" in all_reasons or str(row.get("backtest_verdict", "")).upper() == "FAIL":
        return "Reject for now: historical analogs do not support the setup."
    if "low_sample" in all_reasons:
        return "Watch only: needs more analog history before sizing a trade."
    if "quote replay" in all_reasons:
        if "partial_fail" in all_reasons or "quote replay fail" in all_reasons:
            return "Reject for now: daily option quote replay did not support the setup."
        if "missing_entry" in all_reasons or "missing_exit" in all_reasons or "unavailable" in all_reasons:
            return "Do not trade until both legs can be replayed from daily option snapshots."
        return "Do not trade until daily option quote replay passes."
    if "schwab live failed" in all_reasons:
        if "entry gate miss" in all_reasons:
            return "Do not chase: rerun only if live spread price comes back near modeled entry."
        if "no chain expiry" in all_reasons:
            return "No live chain match: rerun with a listed Schwab expiry before considering entry."
        if "negative spread cost" in all_reasons:
            return "Reject quote: spread pricing is inverted or unusable."
        return "Wait for Schwab live validation to pass."
    if "earnings in trade window" in all_reasons:
        return "Recheck after earnings, or require an explicit earnings-risk trade thesis."
    if "lotto underlying" in all_reasons or "lotto debit" in all_reasons:
        return "Reject for now: low-priced/lotto structure is not a professional-quality trend trade."
    if "flow conflict" in all_reasons:
        return "Reject for now: direction conflicts with options flow."
    if "thin institutional confirmation" in all_reasons:
        return "Watch only: needs stronger whale/institutional confirmation before entry."
    if "directional price not confirming" in all_reasons or "weak directional price trend" in all_reasons:
        return "Watch only: price has to confirm before this can become a directional entry."
    if "directional flow not confirming" in all_reasons or "weak directional flow" in all_reasons:
        return "Watch only: options flow has to strengthen and confirm the trade direction."
    if "expensive debit" in all_reasons:
        return "Do not chase: rebuild the debit spread only if entry cost drops versus max profit."
    if "long strike too far otm" in all_reasons:
        return "Do not chase: wait for the underlying to get closer to the long strike or rebuild nearer the money."
    if "wide market" in all_reasons or "missing bid/ask width" in all_reasons:
        return "Wait for tighter markets; enter only when bid/ask passes the liquidity caps."
    if "short delta" in all_reasons or "short call delta" in all_reasons or "short put delta" in all_reasons:
        return "Reprice farther OTM so every short leg is at or below the max delta gate."
    if "volatile gex" in all_reasons and "iron condor" in strategy:
        return "Skip the condor until GEX stabilizes, or reassess as a directional spread."
    if str(row.get("backtest_verdict", "")).upper() == "PASS":
        return "Near miss: rerun tomorrow and trade only if all live/tradeability gates clear."
    return "Keep on watchlist; not a trade until the blocking gate clears."


def _watchlist_rows(df: pd.DataFrame, *, limit: int = 12) -> List[List[str]]:
    rows: List[List[str]] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        rows.append(
            [
                str(number),
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                str(row.get("strategy", "")),
                _variant_text(row),
                _entry_text(row),
                _fmt_num(row.get("swing_score"), 1),
                _verdict_text(row.get("backtest_verdict", "")),
                _fmt_num(row.get("edge_pct"), 1),
                _watchlist_trigger(row),
            ]
        )
    return rows


def _candidate_rows(df: pd.DataFrame, *, limit: int = DEFAULT_CANDIDATE_TOP) -> List[List[str]]:
    rows: List[List[str]] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        rows.append(
            [
                str(number),
                str(row.get("ticker", "")),
                str(row.get("direction", "")),
                _fmt_num(row.get("swing_score"), 1),
                str(_safe_int(row.get("days_observed"))),
                _clip_text(row.get("candidate_confirmations", ""), 220),
                _clip_text(row.get("candidate_conflicts", ""), 120),
                _clip_text(row.get("candidate_status", ""), 180),
                _clip_text(row.get("candidate_next_step", ""), 160),
            ]
        )
    return rows


def _append_field(lines: List[str], label: str, value: Any, *, limit: int = 320) -> None:
    text = _clip_text(value, limit).replace("\n", " ").strip() or "-"
    lines.append(f"- **{label}:** {text}")


def _candidate_support_text(row: pd.Series) -> str:
    verdict = _verdict_text(row.get("backtest_verdict", ""))
    edge = _fmt_num(row.get("edge_pct"), 1)
    signals = _safe_int(row.get("backtest_signals"))
    variant = _variant_text(row)
    support = str(row.get("candidate_support", "") or "").replace("_", " ").strip()
    if support:
        support = f"{support}; "
    text = f"{support}{verdict}, edge {edge}%, signals {signals}; variant {variant}"
    if "quote_replay_verdict" in row.index:
        qv = str(row.get("quote_replay_verdict", "") or "UNAVAILABLE").strip().upper()
        qpnl = _fmt_money(row.get("quote_replay_pnl"))
        qexit = _clean_cell_text(row.get("quote_replay_exit_date", ""))
        text += f"; quote replay {qv}{' to ' + qexit if qexit else ''}, P&L {qpnl}"
    return text


def _candidate_blocks(df: pd.DataFrame, *, limit: int = DEFAULT_CANDIDATE_TOP) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        score = _fmt_num(row.get("swing_score"), 1)
        days = str(_safe_int(row.get("days_observed")))
        lines.append(f"### {number}. {ticker} - {direction} - score {score}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Window", f"{days} observed market-data days", limit=120)
        _append_field(lines, "Pattern", str(row.get("strategy", "") or "-"), limit=160)
        _append_field(lines, "Backtest support", _candidate_support_text(row), limit=360)
        _append_field(lines, "Confirmations", row.get("candidate_confirmations", ""), limit=420)
        _append_field(lines, "Conflicts", row.get("candidate_conflicts", "") or "none", limit=260)
        _append_field(lines, "Trade readiness", row.get("candidate_status", ""), limit=420)
        _append_field(lines, "Next work", row.get("candidate_next_step", ""), limit=260)
        lines.append("")
    return lines


def _actionable_blocks(
    df: pd.DataFrame,
    *,
    schwab_enabled: bool,
    backtest_enabled: bool,
) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Backtest",
            (
                f"score {_fmt_num(row.get('swing_score'), 1)}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%; "
                f"signals {_safe_int(row.get('backtest_signals'))}"
            ),
            limit=180,
        )
        _append_field(
            lines,
            "Quality gate",
            _gate_text(row, schwab_enabled=schwab_enabled, backtest_enabled=backtest_enabled),
            limit=420,
        )
        _append_field(lines, "Family audit", row.get("strategy_family_summary", ""), limit=360)
        _append_field(lines, "Ticker playbook", row.get("ticker_playbook_summary", ""), limit=360)
        _append_field(lines, "Rolling forward", row.get("rolling_playbook_summary", ""), limit=360)
        _append_field(lines, "Regime", row.get("market_regime_summary", ""), limit=320)
        _append_field(lines, "Open positions", row.get("open_position_summary", "") or "clear", limit=320)
        _append_field(lines, "Position tier", row.get("position_size_tier", ""), limit=160)
        _append_field(lines, "Sizing", row.get("position_size_guidance", ""), limit=260)
        lines.append("")
    return lines


def _max_conviction_blocks(
    df: pd.DataFrame,
    *,
    schwab_enabled: bool,
    backtest_enabled: bool,
) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Position tier", row.get("position_size_tier", ""), limit=120)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Why max conviction",
            (
                f"score {_fmt_num(row.get('swing_score'), 1)}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%; "
                f"signals {_safe_int(row.get('backtest_signals'))}; "
                f"whales {_safe_int(row.get('whale_appearances'))}/{max(1, _safe_int(row.get('days_observed')))}; "
                f"price/flow/OI/DP aligned"
            ),
            limit=360,
        )
        _append_field(
            lines,
            "Quality gate",
            _gate_text(row, schwab_enabled=schwab_enabled, backtest_enabled=backtest_enabled),
            limit=420,
        )
        _append_field(lines, "Sizing", row.get("max_conviction_instruction", ""), limit=260)
        lines.append("")
    return lines


def _trade_workup_blocks(df: pd.DataFrame, *, limit: int = DEFAULT_CANDIDATE_TOP) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Evidence",
            (
                f"score {_fmt_num(row.get('swing_score'), 1)}; "
                f"stock {_fmt_money(row.get('latest_close'))}; "
                f"flow {str(row.get('flow_direction', '-') or '-').lower()}; "
                f"whale days {_safe_int(row.get('whale_appearances'))}"
            ),
            limit=220,
        )
        _append_field(
            lines,
            "Backtest",
            (
                f"{_verdict_text(row.get('backtest_verdict', ''))}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%; "
                f"signals {_safe_int(row.get('backtest_signals'))}"
            ),
            limit=180,
        )
        _append_field(lines, "Why not Actionable Now", row.get("workup_reason", ""), limit=320)
        _append_field(lines, "Next work", row.get("workup_next_step", ""), limit=360)
        _append_field(lines, "Ticker playbook", row.get("ticker_playbook_summary", ""), limit=360)
        lines.append("")
    return lines


def _current_setup_blocks(df: pd.DataFrame, *, limit: int = DEFAULT_CANDIDATE_TOP) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        tier = str(row.get("setup_tier", "") or "SETUP").strip().upper()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {tier} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Entry trigger", row.get("setup_entry_trigger", ""), limit=420)
        _append_field(
            lines,
            "Evidence",
            (
                f"score {_fmt_num(row.get('swing_score'), 1)}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%; "
                f"signals {_safe_int(row.get('backtest_signals'))}; "
                f"price {str(row.get('price_direction', '-') or '-').lower()} "
                f"({_fmt_num(row.get('price_trend'), 1)}); "
                f"flow {str(row.get('flow_direction', '-') or '-').lower()} "
                f"({_fmt_num(row.get('flow_persistence'), 1)})"
            ),
            limit=320,
        )
        _append_field(lines, "Family audit", row.get("strategy_family_summary", ""), limit=360)
        _append_field(lines, "Ticker playbook", row.get("ticker_playbook_summary", ""), limit=360)
        _append_field(lines, "Rolling forward", row.get("rolling_playbook_summary", ""), limit=360)
        _append_field(lines, "Why not order-ready", row.get("setup_reason", ""), limit=420)
        _append_field(lines, "Next step", row.get("setup_next_step", ""), limit=320)
        lines.append("")
    return lines


def _risk_blocked_blocks(df: pd.DataFrame, *, limit: int = 20) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Backtest",
            f"edge {_fmt_num(row.get('edge_pct'), 1)}%; signals {_safe_int(row.get('backtest_signals'))}",
            limit=160,
        )
        _append_field(lines, "Blocked by", row.get("quality_reject_reasons", "") or "-", limit=520)
        lines.append("")
    return lines


def _watchlist_blocks(df: pd.DataFrame, *, limit: int = 12) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Trade setup", _trade_setup_text(row), limit=360)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Evidence",
            (
                f"score {_fmt_num(row.get('swing_score'), 1)}; "
                f"backtest {_verdict_text(row.get('backtest_verdict', ''))}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%"
            ),
            limit=180,
        )
        _append_field(lines, "Trigger before trade", _watchlist_trigger(row), limit=320)
        lines.append("")
    return lines


def _pattern_blocks(df: pd.DataFrame, *, limit: int = 15) -> List[str]:
    lines: List[str] = []
    for number, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        ticker = str(row.get("ticker", "") or "-").strip().upper()
        direction = str(row.get("direction", "") or "-").strip().lower()
        strategy = str(row.get("strategy", "") or "-").strip()
        lines.append(f"### {number}. {ticker} - {direction} - {strategy}")
        _append_field(lines, "Setup tested", _trade_setup_text(row), limit=360)
        _append_field(lines, "Variant", _variant_text(row), limit=260)
        _append_field(
            lines,
            "Backtest",
            (
                f"{_verdict_text(row.get('backtest_verdict', ''))}; "
                f"edge {_fmt_num(row.get('edge_pct'), 1)}%; "
                f"signals {_safe_int(row.get('backtest_signals'))}"
            ),
            limit=180,
        )
        _append_field(lines, "Why not actionable", row.get("actionability_reject_reasons", "") or "-", limit=520)
        _append_field(lines, "Why it showed up", row.get("thesis", ""), limit=520)
        lines.append("")
    return lines


def _file_link(path: Path) -> str:
    return f"[{path.name}]({path})"


def _write_csv_with_fallback_columns(
    df: pd.DataFrame,
    path: Path,
    fallback_columns: Sequence[str],
) -> None:
    if df.empty and len(df.columns) == 0:
        pd.DataFrame(columns=list(fallback_columns)).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def build_report(
    *,
    as_of: dt.date,
    lookback: int,
    trading_days: List[Tuple[dt.date, Path]],
    candidates: pd.DataFrame,
    candidate_shortlist: pd.DataFrame,
    current_setups: pd.DataFrame,
    max_conviction: pd.DataFrame,
    trade_workups: pd.DataFrame,
    actionable: pd.DataFrame,
    patterns: pd.DataFrame,
    walk_forward: pd.DataFrame,
    research_audit: pd.DataFrame,
    research_horizon_audit: pd.DataFrame,
    strategy_family_audit: pd.DataFrame,
    ticker_playbook_audit: pd.DataFrame,
    rolling_ticker_playbook_audit: pd.DataFrame,
    market_regime: Dict[str, Any],
    open_position_summary: Dict[str, Any],
    tracking_summary: Dict[str, Any],
    out_dir: Path,
    raw_report: Path,
    raw_csv: Path,
    candidate_csv: Path,
    current_setups_csv: Path,
    max_conviction_csv: Path,
    trade_workup_csv: Path,
    actionable_csv: Path,
    patterns_csv: Path,
    quote_replay_csv: Path,
    walk_forward_csv: Path,
    research_audit_csv: Path,
    research_horizon_audit_csv: Path,
    research_outcomes_csv: Path,
    strategy_family_audit_csv: Path,
    ticker_playbook_audit_csv: Path,
    rolling_ticker_playbook_audit_csv: Path,
    trade_tracker_csv: Path,
    schwab_enabled: bool,
    backtest_enabled: bool,
    quote_replay_mode: str,
    min_edge: float,
    min_signals: int,
    min_workup_signals: int,
    min_swing_score: float,
    allow_low_sample: bool,
    allow_earnings_risk: bool,
    allow_volatile_ic: bool,
    allow_flow_conflict: bool,
    max_bid_ask_to_price_pct: float,
    max_bid_ask_to_width_pct: float,
    max_short_delta: float,
    min_underlying_price: float,
    min_debit_spread_price: float,
    min_whale_appearances: int,
    walk_forward_requested: bool,
    research_audit_requested: bool,
) -> str:
    lines: List[str] = []
    start = trading_days[0][0].isoformat() if trading_days else ""
    end = trading_days[-1][0].isoformat() if trading_days else as_of.isoformat()
    verdict_counts = (
        candidates.get("backtest_verdict", pd.Series(dtype=str))
        .astype(str)
        .str.upper()
        .replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
        .value_counts()
        .to_dict()
    )
    workup_ids = set()
    if not trade_workups.empty and "_trend_row_id" in trade_workups.columns:
        workup_ids = set(trade_workups["_trend_row_id"].dropna().astype(int).tolist())
    risk_blocked_mask = _truthy_mask(patterns, "base_gate_pass") & ~_truthy_mask(
        patterns, "quality_gate_pass"
    )
    if not patterns.empty:
        risk_blocked_mask &= ~patterns.apply(_has_hard_quality_reject, axis=1)
    if workup_ids and "_trend_row_id" in patterns.columns:
        risk_blocked_mask &= ~patterns["_trend_row_id"].isin(workup_ids)
    risk_blocked_count = int(risk_blocked_mask.sum())

    lines.append(f"# Trend Analysis - {as_of.isoformat()} / L{lookback}")
    lines.append("")
    lines.append(f"- Date window: {start} to {end}")
    lines.append(f"- Effective signal date: {end}")
    lines.append(f"- Trading days analyzed: {len(trading_days)}")
    lines.append(f"- Pattern candidates scored: {len(candidates)}")
    lines.append(f"- Backtest-supported candidates to work on: {len(candidate_shortlist)}")
    lines.append(f"- Current trade setups to work: {len(current_setups)}")
    lines.append(f"- Max-conviction trades: {len(max_conviction)}")
    lines.append(f"- Trade workups: {len(trade_workups)}")
    if "variant_tag" in candidates.columns:
        repair_count = int(candidates["variant_tag"].fillna("base").astype(str).ne("base").sum())
        if repair_count:
            lines.append(f"- Repaired trade variants tested: {repair_count}")
    lines.append(f"- Fully actionable trades: {len(actionable)}")
    lines.append(
        "- Data sources: trends=local UW dated folders; likelihood backtest=yfinance OHLC analogs; "
        "option P&L replay=local UW option snapshots; live validation=Schwab API"
    )
    regime_label = str(market_regime.get("regime", "unknown") or "unknown")
    regime_reason = str(market_regime.get("reason", "") or "").strip()
    lines.append(
        f"- Market regime filter: {regime_label}"
        + (f" ({regime_reason})" if regime_reason else "")
    )
    open_status = "skipped" if open_position_summary.get("skipped") else (
        "checked" if open_position_summary.get("checked") else "not checked"
    )
    lines.append(
        "- Open-position awareness: "
        + f"{open_status}; open underlyings={int(open_position_summary.get('open_underlyings', 0) or 0)}; "
        + f"blocked rows={int(open_position_summary.get('blocked_rows', 0) or 0)}"
    )
    if not rolling_ticker_playbook_audit.empty:
        rolling_counts = (
            rolling_ticker_playbook_audit.get("verdict", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        rolling_text = ", ".join(f"{k}={v}" for k, v in sorted(rolling_counts.items()))
        lines.append(f"- Rolling ticker-playbook forward validation: {rolling_text}")
    else:
        lines.append("- Rolling ticker-playbook forward validation: no completed rows")
    if tracking_summary.get("enabled"):
        outcome_summary = tracking_summary.get("outcomes", {}) if isinstance(tracking_summary.get("outcomes"), dict) else {}
        lines.append(
            "- Post-trade tracking: "
            + f"added {int(tracking_summary.get('added', 0) or 0)}, "
            + f"updated {int(tracking_summary.get('updated', 0) or 0)}, "
            + f"outcomes refreshed {int(outcome_summary.get('updated', 0) or 0)}, "
            + f"wins/losses {int(outcome_summary.get('wins', 0) or 0)}/{int(outcome_summary.get('losses', 0) or 0)}, "
            + f"total {int(tracking_summary.get('total', 0) or 0)}"
        )
    else:
        lines.append("- Post-trade tracking: skipped")
    if risk_blocked_count:
        base_gate_label = "backtest/Schwab" if schwab_enabled else "backtest"
        lines.append(f"- Risk-blocked {base_gate_label} passes: {risk_blocked_count}")
    lines.append(
        f"- Backtest gate: PASS, edge >= {min_edge:.1f}%, signals >= {int(min_signals)}"
        + ("; LOW_SAMPLE allowed" if allow_low_sample else "")
        if backtest_enabled
        else "- Backtest gate: skipped, no trades are marked actionable"
    )
    lines.append(
        f"- Trade Workup gate: quality/live/quote pass, edge >= {min_edge:.1f}%, signals >= {int(min_workup_signals)}, but below Actionable Now sample support"
    )
    if float(min_swing_score) > 0:
        lines.append(f"- Trend evidence gate: swing score >= {float(min_swing_score):.1f}")
    lines.append(
        "- Professional-quality gate: "
        + f"underlying >= ${float(min_underlying_price):.2f}"
        + f"; debit spreads >= ${float(min_debit_spread_price):.2f}"
        + f"; whale days scale by lookback, capped at {int(min_whale_appearances)}"
        + f"; directional debit price/flow scores >= {DEFAULT_MIN_DIRECTIONAL_PRICE_SCORE:.0f}/{DEFAULT_MIN_DIRECTIONAL_FLOW_SCORE:.0f}"
        + f"; debit <= {DEFAULT_MAX_DEBIT_TO_WIDTH_PCT:.0%} of width"
        + f"; long strike <= {DEFAULT_MAX_LONG_STRIKE_OTM_PCT:.0%} OTM"
        + ("; flow conflicts allowed" if allow_flow_conflict else "; no directional flow conflict")
    )
    lines.append(f"- Schwab live validation: {'enabled' if schwab_enabled else 'skipped'}")
    if schwab_enabled:
        lines.append(
            "- Tradeability gate: "
            + ("earnings allowed" if allow_earnings_risk else "no earnings through expiry")
            + f"; bid/ask <= {max_bid_ask_to_price_pct:.0%} of spread price"
            + f"; bid/ask <= {max_bid_ask_to_width_pct:.0%} of spread width"
            + f"; short delta <= {max_short_delta:.2f}"
            + ("; volatile IC allowed" if allow_volatile_ic else "; no volatile-GEX iron condors")
        )
    if verdict_counts:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(verdict_counts.items()))
        lines.append(f"- Backtest verdict mix: {counts}")
    quote_counts = trend_quote_replay.quote_replay_summary(candidates)
    if str(quote_replay_mode or "off").lower() == "off":
        lines.append("- Daily option quote replay: off")
    else:
        replay_label = "gate" if str(quote_replay_mode).lower() == "gate" else "diagnostic"
        counts = ", ".join(f"{k}={v}" for k, v in sorted(quote_counts.items())) if quote_counts else "none"
        lines.append(f"- Daily option quote replay: {replay_label}; {counts}")
    if walk_forward_requested:
        lines.append(f"- Walk-forward audit: {_walk_forward_confidence_text(walk_forward)}")
    else:
        lines.append("- Walk-forward audit: skipped")
    if research_audit_requested and not research_audit.empty:
        verdicts = research_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str)
        if verdicts.eq("supportive").any():
            lines.append("- Research confidence audit: at least one fixed bucket is supportive")
        else:
            lines.append("- Research confidence audit: no fixed bucket is supportive")
        if not research_horizon_audit.empty:
            horizon_verdicts = research_horizon_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str)
            if horizon_verdicts.eq("supportive").any():
                lines.append("- Research horizon audit: at least one holding horizon is supportive")
            else:
                lines.append("- Research horizon audit: no holding horizon is supportive")
        if not strategy_family_audit.empty:
            family_verdicts = strategy_family_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str)
            if family_verdicts.eq("promotable").any():
                lines.append("- Strategy family audit: at least one setup family is promotable")
            else:
                lines.append("- Strategy family audit: no setup family is promotable")
        if not ticker_playbook_audit.empty:
            playbook_verdicts = ticker_playbook_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str)
            if playbook_verdicts.eq("promotable").any():
                lines.append("- Ticker playbook audit: at least one ticker-specific setup is promotable")
            else:
                lines.append("- Ticker playbook audit: no ticker-specific setup is promotable")
    elif research_audit_requested:
        lines.append("- Research confidence audit: no completed historical bucket outcomes")
    else:
        lines.append("- Research confidence audit: skipped")
    lines.append("")

    lines.append("## Walk-Forward Audit")
    if walk_forward_requested:
        lines.extend(_walk_forward_report_lines(walk_forward))
    else:
        lines.append("_skipped for this run; use `--walk-forward-samples N` to run the historical outcome audit_")
    lines.append("")

    lines.append("## Research Confidence Audit")
    if research_audit_requested:
        lines.extend(_research_audit_report_lines(research_audit))
    else:
        lines.append("_skipped because walk-forward audit was skipped_")
    lines.append("")

    lines.append("## Research Horizon Audit")
    if research_audit_requested:
        lines.extend(_research_horizon_report_lines(research_horizon_audit))
    else:
        lines.append("_skipped because walk-forward audit was skipped_")
    lines.append("")

    lines.append("## Strategy Family Audit")
    if research_audit_requested:
        lines.extend(_strategy_family_report_lines(strategy_family_audit))
    else:
        lines.append("_skipped because walk-forward audit was skipped_")
    lines.append("")

    lines.append("## Ticker Playbook Audit")
    if research_audit_requested:
        lines.extend(_ticker_playbook_report_lines(ticker_playbook_audit))
    else:
        lines.append("_skipped because walk-forward audit was skipped_")
    lines.append("")

    lines.append("## Rolling Ticker Playbook Forward Validation")
    if research_audit_requested:
        lines.extend(_rolling_ticker_playbook_report_lines(rolling_ticker_playbook_audit))
    else:
        lines.append("_skipped because walk-forward audit was skipped_")
    lines.append("")

    lines.append("## Current Trade Setups")
    if current_setups.empty:
        lines.append(
            "No current setup is clean enough to work. That means the pipeline found patterns, but none have a usable conditional setup after live, quote, and professional-quality filters."
        )
    else:
        lines.append(
            "These are the best current setups to work. They are not automatically order tickets; the setup tier and entry trigger say what must be true before entry."
        )
        lines.append("")
        lines.extend(_current_setup_blocks(current_setups))
    lines.append("")

    lines.append("## Backtest-Supported Candidate Shortlist")
    if candidate_shortlist.empty:
        lines.append(
            "No candidate cleared both the trend-evidence gate and the backtest-support gate. That is acceptable; do not force a trade from weak evidence."
        )
    else:
        lines.append(
            "These are the names worth working on first because the trend evidence has at least one backtest-supported structure. They are not automatically trades; trade readiness is listed separately."
        )
        lines.append("")
        lines.extend(_candidate_blocks(candidate_shortlist))
    lines.append("")

    lines.append("## Actionable Trades")
    if actionable.empty:
        lines.append(
            "No trades passed the full backtest/live/tradeability gate. Use Trade Workup for quality setups that need confirmation; do not treat them as entries."
        )
    else:
        lines.extend(
            _actionable_blocks(
                actionable,
                schwab_enabled=schwab_enabled,
                backtest_enabled=backtest_enabled,
            )
        )
    lines.append("")

    lines.append("## Max Conviction / Max Planned Risk")
    if max_conviction.empty:
        lines.append("_none_")
    else:
        lines.append(
            "These pass Actionable Now plus stricter alignment and liquidity checks. This means max pre-defined risk for one defined-risk trade, not all account capital."
        )
        lines.append("")
        lines.extend(
            _max_conviction_blocks(
                max_conviction,
                schwab_enabled=schwab_enabled,
                backtest_enabled=backtest_enabled,
            )
        )
    lines.append("")

    lines.append("## Trade Workup")
    if trade_workups.empty:
        lines.append("_none_")
    else:
        lines.append(
            "These are quality trend setups worth working on, but they are not order tickets. They pass the professional, live, and quote gates, then stop short on sample support."
        )
        lines.append("")
        lines.extend(_trade_workup_blocks(trade_workups))
    lines.append("")

    risk_blocked = patterns[risk_blocked_mask].copy()
    excluded_pattern_ids = set()
    if "_trend_row_id" in risk_blocked.columns:
        excluded_pattern_ids.update(risk_blocked["_trend_row_id"].dropna().astype(int).tolist())
    excluded_pattern_ids.update(workup_ids)
    pattern_only = patterns.copy()
    if excluded_pattern_ids and "_trend_row_id" in pattern_only.columns:
        pattern_only = pattern_only[~pattern_only["_trend_row_id"].isin(excluded_pattern_ids)].copy()
    else:
        pattern_only = pattern_only.drop(index=risk_blocked.index, errors="ignore").copy()

    lines.append("## Risk-Blocked Backtest / Schwab Passes")
    if risk_blocked.empty:
        lines.append("_none_")
    else:
        lines.append(
            "These passed the base backtest/live gates but were blocked by tradeability risk."
        )
        lines.append("")
        lines.extend(_risk_blocked_blocks(risk_blocked))
    lines.append("")

    watchlist_source = pd.concat(
        [risk_blocked, trade_workups, pattern_only],
        ignore_index=True,
    )
    watchlist_source = _dedupe_trade_rows(watchlist_source, by_ticker=True).head(12)
    lines.append("## Watchlist Triggers")
    if watchlist_source.empty:
        lines.append("_none_")
    else:
        lines.append(
            "These are not trade recommendations. They are the best setups to recheck, with the condition that has to change before entry."
        )
        lines.append("")
        lines.extend(_watchlist_blocks(watchlist_source))
    lines.append("")

    lines.append("## Pattern Candidates")
    if pattern_only.empty:
        lines.append("_none_")
    else:
        lines.append(
            "These had trend evidence but did not pass the base backtest/live gates."
        )
        lines.append("")
        lines.extend(_pattern_blocks(pattern_only))
    lines.append("")

    lines.append("## Post-Trade Tracking")
    if tracking_summary.get("enabled"):
        outcome_summary = tracking_summary.get("outcomes", {}) if isinstance(tracking_summary.get("outcomes"), dict) else {}
        lines.append(
            f"Tracking CSV: {_file_link(trade_tracker_csv)}. "
            f"Added {int(tracking_summary.get('added', 0) or 0)} new row(s), "
            f"updated {int(tracking_summary.get('updated', 0) or 0)} existing row(s), "
            f"refreshed outcomes for {int(outcome_summary.get('updated', 0) or 0)} row(s), "
            f"scored {int(outcome_summary.get('scored', 0) or 0)}, "
            f"unavailable {int(outcome_summary.get('unavailable', 0) or 0)}, "
            f"total tracked rows {int(tracking_summary.get('total', 0) or 0)}."
        )
    else:
        lines.append("_skipped for this run_")
    lines.append("")

    lines.append("## Files")
    report_path = out_dir / f"trend-analysis-{as_of.isoformat()}-L{lookback}.md"
    lines.append(f"- This report: {_file_link(report_path)}")
    lines.append(f"- Candidate CSV: {_file_link(candidate_csv)}")
    lines.append(f"- Current Setups CSV: {_file_link(current_setups_csv)}")
    lines.append(f"- Max Conviction CSV: {_file_link(max_conviction_csv)}")
    lines.append(f"- Trade Workup CSV: {_file_link(trade_workup_csv)}")
    lines.append(f"- Actionable CSV: {_file_link(actionable_csv)}")
    lines.append(f"- Pattern CSV: {_file_link(patterns_csv)}")
    if str(quote_replay_mode or "off").lower() != "off":
        lines.append(f"- Quote replay CSV: {_file_link(quote_replay_csv)}")
    if walk_forward_requested:
        lines.append(f"- Walk-forward CSV: {_file_link(walk_forward_csv)}")
    if research_audit_requested:
        lines.append(f"- Research audit CSV: {_file_link(research_audit_csv)}")
        lines.append(f"- Research horizon audit CSV: {_file_link(research_horizon_audit_csv)}")
        lines.append(f"- Research outcomes CSV: {_file_link(research_outcomes_csv)}")
        lines.append(f"- Strategy family audit CSV: {_file_link(strategy_family_audit_csv)}")
        lines.append(f"- Ticker playbook audit CSV: {_file_link(ticker_playbook_audit_csv)}")
        lines.append(f"- Rolling ticker playbook audit CSV: {_file_link(rolling_ticker_playbook_audit_csv)}")
    if tracking_summary.get("enabled"):
        lines.append(f"- Post-trade tracker CSV: {_file_link(trade_tracker_csv)}")
    lines.append(f"- Raw swing report: {_file_link(raw_report)}")
    lines.append(f"- Raw swing CSV: {_file_link(raw_csv)}")
    return "\n".join(lines)


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to a mapping: {config_path}")
    return cfg


def _output_names(as_of: dt.date, lookback: int) -> Dict[str, str]:
    suffix = f"{as_of.isoformat()}-L{lookback}"
    return {
        "report": f"trend-analysis-{suffix}.md",
        "candidate_csv": f"trend-analysis-candidates-{suffix}.csv",
        "current_setups_csv": f"trend-analysis-current-setups-{suffix}.csv",
        "max_conviction_csv": f"trend-analysis-max-conviction-{suffix}.csv",
        "trade_workup_csv": f"trend-analysis-trade-workups-{suffix}.csv",
        "actionable_csv": f"trend-analysis-actionable-{suffix}.csv",
        "patterns_csv": f"trend-analysis-patterns-{suffix}.csv",
        "quote_replay_csv": f"trend-analysis-quote-replay-{suffix}.csv",
        "walk_forward_csv": f"trend-analysis-walk-forward-{suffix}.csv",
        "research_audit_csv": f"trend-analysis-research-audit-{suffix}.csv",
        "research_horizon_audit_csv": f"trend-analysis-research-audit-by-horizon-{suffix}.csv",
        "research_outcomes_csv": f"trend-analysis-research-outcomes-{suffix}.csv",
        "strategy_family_audit_csv": f"trend-analysis-strategy-family-audit-{suffix}.csv",
        "ticker_playbook_audit_csv": f"trend-analysis-ticker-playbook-audit-{suffix}.csv",
        "rolling_ticker_playbook_audit_csv": f"trend-analysis-rolling-ticker-playbook-audit-{suffix}.csv",
        "metadata": f"trend-analysis-metadata-{suffix}.json",
        "raw_report": f"trend-analysis-raw-{as_of.isoformat()}-L{lookback}.md",
        "raw_csv": f"trend_analysis_raw_{as_of.isoformat()}-L{lookback}.csv",
    }


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    args = parse_args(argv)
    as_of, lookback = resolve_invocation(args)
    root = Path(args.root_dir).expanduser().resolve() if args.root_dir else default_root_dir()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (root / "out" / "trend_analysis").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if as_of is None:
        days = swing.discover_trading_days(root, lookback, None)
        if not days:
            raise RuntimeError(f"No dated folders discovered under {root}")
        as_of = days[-1][0]
    trading_days = swing.discover_trading_days(root, lookback, as_of)
    if not trading_days:
        raise RuntimeError(f"No dated folders discovered under {root} on or before {as_of}")

    cfg = _load_config(Path(args.config).expanduser().resolve())
    names = _output_names(as_of, lookback)
    cfg.setdefault("pipeline", {})["root_dir"] = str(root)
    cfg.setdefault("pipeline", {})["lookback_days"] = lookback
    cfg.setdefault("pipeline", {})["output_dir"] = str(out_dir)
    cfg.setdefault("output", {})["report_md_name"] = names["raw_report"].replace(f"-L{lookback}.md", ".md")
    cfg.setdefault("output", {})["shortlist_csv_name"] = names["raw_csv"].replace(f"-L{lookback}.csv", ".csv")
    cfg.setdefault("schwab_validation", {})["enabled"] = not bool(args.no_schwab)
    cfg.setdefault("backtest", {})["enabled"] = not bool(args.no_backtest)
    cfg.setdefault("backtest", {})["min_signals"] = int(args.min_backtest_signals)
    cfg.setdefault("backtest", {})["max_setups"] = int(args.max_backtest_setups)
    if args.cache_dir:
        cfg.setdefault("backtest", {})["cache_dir"] = args.cache_dir

    top_n = int(args.top)
    candidate_pool = int(args.candidate_pool or max(top_n * 3, top_n + 10))

    print("Trend Analysis Pipeline", flush=True)
    print(f"  Root: {root}", flush=True)
    print(f"  As-of: {as_of.isoformat()}", flush=True)
    print(f"  Lookback: {lookback} trading days", flush=True)
    print(f"  Candidate pool: {candidate_pool}", flush=True)
    print(f"  Backtest: {'enabled' if not args.no_backtest else 'skipped'}", flush=True)
    print(f"  Schwab: {'enabled' if not args.no_schwab else 'skipped'}", flush=True)

    raw_report = out_dir / names["raw_report"]
    raw_csv = out_dir / names["raw_csv"]
    candidate_csv = out_dir / names["candidate_csv"]
    current_setups_csv = out_dir / names["current_setups_csv"]
    max_conviction_csv = out_dir / names["max_conviction_csv"]
    trade_workup_csv = out_dir / names["trade_workup_csv"]
    actionable_csv = out_dir / names["actionable_csv"]
    patterns_csv = out_dir / names["patterns_csv"]
    quote_replay_csv = out_dir / names["quote_replay_csv"]
    walk_forward_csv = out_dir / names["walk_forward_csv"]
    research_audit_csv = out_dir / names["research_audit_csv"]
    research_horizon_audit_csv = out_dir / names["research_horizon_audit_csv"]
    research_outcomes_csv = out_dir / names["research_outcomes_csv"]
    strategy_family_audit_csv = out_dir / names["strategy_family_audit_csv"]
    ticker_playbook_audit_csv = out_dir / names["ticker_playbook_audit_csv"]
    rolling_ticker_playbook_audit_csv = out_dir / names["rolling_ticker_playbook_audit_csv"]
    trade_tracker_csv = (
        Path(args.trade_tracker).expanduser().resolve()
        if args.trade_tracker
        else out_dir / DEFAULT_TRACKING_FILE_NAME
    )
    report_path = out_dir / names["report"]
    metadata_path = out_dir / names["metadata"]

    scores, signals = swing.run_pipeline(
        cfg=cfg,
        root=root,
        lookback=lookback,
        as_of=as_of,
        out_dir=out_dir,
        max_recommendations=candidate_pool,
    )
    if raw_csv.exists():
        candidates = pd.read_csv(raw_csv, low_memory=False)
    else:
        candidates = swing.generate_shortlist_csv(scores, signals)

    quote_replay_mode = str(args.quote_replay or "off").strip().lower()
    quote_replay_results = pd.DataFrame()
    if quote_replay_mode != "off":
        effective_signal_date = trading_days[-1][0]
        print(
            f"  Quote replay: {quote_replay_mode} from {effective_signal_date.isoformat()}",
            flush=True,
        )
        candidates, quote_replay_results = trend_quote_replay.annotate_quote_replay(
            candidates,
            root=root,
            signal_date=effective_signal_date,
            mode=quote_replay_mode,
            allow_web_fallback=bool(args.quote_replay_web_fallback),
        )
        candidates.to_csv(quote_replay_csv, index=False)

    if bool(args.no_regime_filter):
        market_regime: Dict[str, Any] = {"regime": "skipped", "reason": "regime filter skipped"}
        candidates = candidates.copy()
        candidates["market_regime"] = "skipped"
        candidates["market_regime_gate_pass"] = True
        candidates["market_regime_summary"] = "regime filter skipped"
    else:
        market_regime = compute_market_regime(root, trading_days)
        candidates = annotate_regime_filter(candidates, market_regime)

    if bool(args.no_position_check):
        open_position_summary: Dict[str, Any] = {
            "checked": False,
            "skipped": True,
            "position_json": "",
            "open_underlyings": 0,
            "blocked_rows": 0,
        }
        candidates = candidates.copy()
        candidates["open_position_gate_pass"] = True
        candidates["open_position_status"] = "skipped"
        candidates["open_position_summary"] = "position check skipped"
    else:
        position_json = (
            Path(args.position_json).expanduser().resolve()
            if args.position_json
            else _latest_position_json(root)
        )
        candidates, open_position_summary = annotate_open_position_awareness(candidates, position_json)

    actionable, patterns = split_actionable_candidates(
        candidates,
        top_n=top_n,
        backtest_enabled=not bool(args.no_backtest),
        schwab_enabled=not bool(args.no_schwab),
        quote_replay_mode=quote_replay_mode,
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_swing_score=float(args.candidate_min_score),
        allow_low_sample=bool(args.allow_low_sample),
        allow_earnings_risk=bool(args.allow_earnings_risk),
        allow_volatile_ic=bool(args.allow_volatile_ic),
        allow_flow_conflict=bool(args.allow_flow_conflict),
        max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
        max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
        max_short_delta=float(args.max_short_delta),
        min_underlying_price=float(args.min_underlying_price),
        min_debit_spread_price=float(args.min_debit_spread_price),
        min_whale_appearances=int(args.min_whale_appearances),
    )
    annotated_candidates = pd.concat([actionable, patterns], ignore_index=True)
    candidate_shortlist = build_candidate_shortlist(
        annotated_candidates,
        top_n=int(args.candidate_top),
        min_score=float(args.candidate_min_score),
        min_confirmations=int(args.candidate_min_confirmations),
        max_conflicts=int(args.candidate_max_conflicts),
    )
    shortlist_tickers = (
        set(candidate_shortlist.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper())
        if not candidate_shortlist.empty
        else set()
    )
    trade_workups = build_trade_workups(
        patterns,
        top_n=int(args.candidate_top),
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_workup_signals=int(args.min_workup_signals),
        min_swing_score=float(args.candidate_min_score),
        exclude_tickers=shortlist_tickers,
    )
    current_setups = build_current_trade_setups(
        annotated_candidates,
        top_n=int(args.candidate_top),
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_workup_signals=int(args.min_workup_signals),
        min_swing_score=float(args.candidate_min_score),
    )
    actionable = annotate_position_sizing(actionable)
    max_conviction = build_max_conviction(actionable, top_n=top_n)
    if bool(args.reuse_walk_forward_outcomes) and walk_forward_csv.exists():
        walk_forward = pd.read_csv(walk_forward_csv, low_memory=False)
    else:
        walk_forward = run_walk_forward_audit(
            root=root,
            out_dir=out_dir,
            cfg_template=cfg,
            as_of=as_of,
            lookback=lookback,
            candidate_pool=candidate_pool,
            top_n=int(args.walk_forward_top),
            samples=max(0, int(args.walk_forward_samples)),
            horizons=_parse_walk_forward_horizons(args.walk_forward_horizons),
            cache_dir=str(args.cache_dir or ""),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_swing_score=float(args.candidate_min_score),
            allow_low_sample=bool(args.allow_low_sample),
            allow_earnings_risk=bool(args.allow_earnings_risk),
            allow_volatile_ic=bool(args.allow_volatile_ic),
            allow_flow_conflict=bool(args.allow_flow_conflict),
            max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
            max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
            max_short_delta=float(args.max_short_delta),
            min_underlying_price=float(args.min_underlying_price),
            min_debit_spread_price=float(args.min_debit_spread_price),
            min_whale_appearances=int(args.min_whale_appearances),
            reuse_raw=bool(args.reuse_walk_forward_raw),
        )
    if bool(args.reuse_research_outcomes) and research_outcomes_csv.exists():
        research_outcomes = pd.read_csv(research_outcomes_csv, low_memory=False)
    else:
        research_outcomes = collect_research_confidence_outcomes(
            root=root,
            out_dir=out_dir,
            as_of=as_of,
            lookback=lookback,
            samples=max(0, int(args.walk_forward_samples)),
            horizons=_parse_walk_forward_horizons(args.walk_forward_horizons),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_workup_signals=int(args.min_workup_signals),
            min_swing_score=float(args.candidate_min_score),
            allow_low_sample=bool(args.allow_low_sample),
            allow_earnings_risk=bool(args.allow_earnings_risk),
            allow_volatile_ic=bool(args.allow_volatile_ic),
            allow_flow_conflict=bool(args.allow_flow_conflict),
            max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
            max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
            max_short_delta=float(args.max_short_delta),
            min_underlying_price=float(args.min_underlying_price),
            min_debit_spread_price=float(args.min_debit_spread_price),
            min_whale_appearances=int(args.min_whale_appearances),
        )
    research_audit = _research_summary_from_outcomes(research_outcomes)
    research_horizon_audit = _research_summary_by_horizon_from_outcomes(research_outcomes)
    strategy_family_audit = _strategy_family_audit_from_outcomes(research_outcomes)
    ticker_playbook_audit = _ticker_playbook_audit_from_outcomes(research_outcomes)
    rolling_ticker_playbook_audit = _rolling_ticker_playbook_audit_from_outcomes(
        research_outcomes,
        ticker_playbook_audit,
    )
    research_requested = max(0, int(args.walk_forward_samples)) > 0
    if research_requested:
        candidates = annotate_strategy_family_gate(candidates, strategy_family_audit)
        candidates = annotate_ticker_playbook_gate(candidates, ticker_playbook_audit)
        candidates = annotate_rolling_playbook_gate(candidates, rolling_ticker_playbook_audit)
        actionable, patterns = split_actionable_candidates(
            candidates,
            top_n=top_n,
            backtest_enabled=not bool(args.no_backtest),
            schwab_enabled=not bool(args.no_schwab),
            quote_replay_mode=quote_replay_mode,
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_swing_score=float(args.candidate_min_score),
            allow_low_sample=bool(args.allow_low_sample),
            allow_earnings_risk=bool(args.allow_earnings_risk),
            allow_volatile_ic=bool(args.allow_volatile_ic),
            allow_flow_conflict=bool(args.allow_flow_conflict),
            max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
            max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
            max_short_delta=float(args.max_short_delta),
            min_underlying_price=float(args.min_underlying_price),
            min_debit_spread_price=float(args.min_debit_spread_price),
            min_whale_appearances=int(args.min_whale_appearances),
        )
        annotated_candidates = pd.concat([actionable, patterns], ignore_index=True)
        candidate_shortlist = build_candidate_shortlist(
            annotated_candidates,
            top_n=int(args.candidate_top),
            min_score=float(args.candidate_min_score),
            min_confirmations=int(args.candidate_min_confirmations),
            max_conflicts=int(args.candidate_max_conflicts),
        )
        shortlist_tickers = (
            set(candidate_shortlist.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper())
            if not candidate_shortlist.empty
            else set()
        )
        trade_workups = build_trade_workups(
            patterns,
            top_n=int(args.candidate_top),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_workup_signals=int(args.min_workup_signals),
            min_swing_score=float(args.candidate_min_score),
            exclude_tickers=shortlist_tickers,
        )
        current_setups = build_current_trade_setups(
            annotated_candidates,
            top_n=int(args.candidate_top),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_workup_signals=int(args.min_workup_signals),
            min_swing_score=float(args.candidate_min_score),
        )
        actionable = annotate_position_sizing(actionable)
        max_conviction = build_max_conviction(actionable, top_n=top_n)
    if (
        research_requested
        and not (
            _research_confidence_supportive(research_audit, research_horizon_audit)
            or _strategy_family_supportive(strategy_family_audit)
            or _ticker_playbook_supportive(ticker_playbook_audit)
        )
        and not actionable.empty
    ):
        reason = "research confidence/strategy family audits have no supportive or promotable bucket"
        blocked = actionable.copy()
        prior = blocked.get("actionability_reject_reasons", pd.Series("", index=blocked.index)).fillna("").astype(str)
        blocked["actionability_reject_reasons"] = np.where(
            prior.str.strip().ne(""),
            prior + "; " + reason,
            reason,
        )
        base_prior = blocked.get("base_gate_reasons", pd.Series("", index=blocked.index)).fillna("").astype(str)
        blocked["base_gate_reasons"] = np.where(
            base_prior.str.strip().ne(""),
            base_prior + "; " + reason,
            reason,
        )
        blocked["base_gate_pass"] = False
        blocked["quality_gate_pass"] = blocked.get("quality_gate_pass", pd.Series(True, index=blocked.index))
        patterns = pd.concat([blocked, patterns], ignore_index=True)
        actionable = actionable.iloc[0:0].copy()
        annotated_candidates = pd.concat([actionable, patterns], ignore_index=True)
        candidate_shortlist = build_candidate_shortlist(
            annotated_candidates,
            top_n=int(args.candidate_top),
            min_score=float(args.candidate_min_score),
            min_confirmations=int(args.candidate_min_confirmations),
            max_conflicts=int(args.candidate_max_conflicts),
        )
        shortlist_tickers = (
            set(candidate_shortlist.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper())
            if not candidate_shortlist.empty
            else set()
        )
        trade_workups = build_trade_workups(
            patterns,
            top_n=int(args.candidate_top),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_workup_signals=int(args.min_workup_signals),
            min_swing_score=float(args.candidate_min_score),
            exclude_tickers=shortlist_tickers,
        )
        current_setups = build_current_trade_setups(
            annotated_candidates,
            top_n=int(args.candidate_top),
            min_edge=float(args.min_backtest_edge),
            min_signals=int(args.min_backtest_signals),
            min_workup_signals=int(args.min_workup_signals),
            min_swing_score=float(args.candidate_min_score),
        )
        max_conviction = build_max_conviction(actionable, top_n=top_n)

    actionable = annotate_position_sizing(actionable)
    max_conviction = build_max_conviction(actionable, top_n=top_n)
    tracking_summary = update_trade_tracking(
        actionable,
        trade_tracker_csv,
        report_path=report_path,
        as_of=as_of,
        enabled=not bool(args.no_trade_tracking),
    )
    outcome_summary = refresh_trade_tracking_outcomes(
        trade_tracker_csv,
        root=root,
        as_of=as_of,
        enabled=not bool(args.no_trade_tracking) and not bool(args.no_outcome_update),
    )
    tracking_summary["outcomes"] = outcome_summary

    if str(quote_replay_mode or "off").lower() != "off":
        candidates.to_csv(quote_replay_csv, index=False)
    fallback_candidate_columns = list(candidates.columns)
    _write_csv_with_fallback_columns(candidate_shortlist, candidate_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(current_setups, current_setups_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(max_conviction, max_conviction_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(trade_workups, trade_workup_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(actionable, actionable_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(patterns, patterns_csv, fallback_candidate_columns)
    _write_csv_with_fallback_columns(walk_forward, walk_forward_csv, WALK_FORWARD_COLUMNS)
    _write_csv_with_fallback_columns(research_audit, research_audit_csv, RESEARCH_AUDIT_COLUMNS)
    _write_csv_with_fallback_columns(
        research_horizon_audit,
        research_horizon_audit_csv,
        RESEARCH_HORIZON_AUDIT_COLUMNS,
    )
    _write_csv_with_fallback_columns(research_outcomes, research_outcomes_csv, RESEARCH_OUTCOME_COLUMNS)
    _write_csv_with_fallback_columns(
        strategy_family_audit,
        strategy_family_audit_csv,
        STRATEGY_FAMILY_AUDIT_COLUMNS,
    )
    _write_csv_with_fallback_columns(
        ticker_playbook_audit,
        ticker_playbook_audit_csv,
        TICKER_PLAYBOOK_AUDIT_COLUMNS,
    )
    _write_csv_with_fallback_columns(
        rolling_ticker_playbook_audit,
        rolling_ticker_playbook_audit_csv,
        ROLLING_TICKER_PLAYBOOK_AUDIT_COLUMNS,
    )
    report_text = build_report(
        as_of=as_of,
        lookback=lookback,
        trading_days=trading_days,
        candidates=candidates,
        candidate_shortlist=candidate_shortlist,
        current_setups=current_setups,
        max_conviction=max_conviction,
        trade_workups=trade_workups,
        actionable=actionable,
        patterns=patterns,
        walk_forward=walk_forward,
        research_audit=research_audit,
        research_horizon_audit=research_horizon_audit,
        strategy_family_audit=strategy_family_audit,
        ticker_playbook_audit=ticker_playbook_audit,
        rolling_ticker_playbook_audit=rolling_ticker_playbook_audit,
        market_regime=market_regime,
        open_position_summary=open_position_summary,
        tracking_summary=tracking_summary,
        out_dir=out_dir,
        raw_report=raw_report,
        raw_csv=raw_csv,
        candidate_csv=candidate_csv,
        current_setups_csv=current_setups_csv,
        max_conviction_csv=max_conviction_csv,
        trade_workup_csv=trade_workup_csv,
        actionable_csv=actionable_csv,
        patterns_csv=patterns_csv,
        quote_replay_csv=quote_replay_csv,
        walk_forward_csv=walk_forward_csv,
        research_audit_csv=research_audit_csv,
        research_horizon_audit_csv=research_horizon_audit_csv,
        research_outcomes_csv=research_outcomes_csv,
        strategy_family_audit_csv=strategy_family_audit_csv,
        ticker_playbook_audit_csv=ticker_playbook_audit_csv,
        rolling_ticker_playbook_audit_csv=rolling_ticker_playbook_audit_csv,
        trade_tracker_csv=trade_tracker_csv,
        schwab_enabled=not bool(args.no_schwab),
        backtest_enabled=not bool(args.no_backtest),
        quote_replay_mode=quote_replay_mode,
        min_edge=float(args.min_backtest_edge),
        min_signals=int(args.min_backtest_signals),
        min_workup_signals=int(args.min_workup_signals),
        min_swing_score=float(args.candidate_min_score),
        allow_low_sample=bool(args.allow_low_sample),
        allow_earnings_risk=bool(args.allow_earnings_risk),
        allow_volatile_ic=bool(args.allow_volatile_ic),
        allow_flow_conflict=bool(args.allow_flow_conflict),
        max_bid_ask_to_price_pct=float(args.max_bid_ask_to_price_pct),
        max_bid_ask_to_width_pct=float(args.max_bid_ask_to_width_pct),
        max_short_delta=float(args.max_short_delta),
        min_underlying_price=float(args.min_underlying_price),
        min_debit_spread_price=float(args.min_debit_spread_price),
        min_whale_appearances=int(args.min_whale_appearances),
        walk_forward_requested=max(0, int(args.walk_forward_samples)) > 0,
        research_audit_requested=max(0, int(args.walk_forward_samples)) > 0,
    )
    report_path.write_text(report_text, encoding="utf-8")

    metadata = {
        "root_dir": str(root),
        "out_dir": str(out_dir),
        "as_of": as_of.isoformat(),
        "effective_signal_date": trading_days[-1][0].isoformat() if trading_days else as_of.isoformat(),
        "lookback": lookback,
        "trading_days": [d.isoformat() for d, _ in trading_days],
        "candidate_pool": candidate_pool,
        "candidates": int(len(candidates)),
        "candidate_shortlist": int(len(candidate_shortlist)),
        "current_setups": int(len(current_setups)),
        "max_conviction": int(len(max_conviction)),
        "trade_workups": int(len(trade_workups)),
        "actionable": int(len(actionable)),
        "patterns": int(len(patterns)),
        "backtest_enabled": not bool(args.no_backtest),
        "schwab_enabled": not bool(args.no_schwab),
        "quote_replay_mode": quote_replay_mode,
        "quote_replay_counts": trend_quote_replay.quote_replay_summary(candidates),
        "quote_replay_rows": int(len(quote_replay_results)),
        "walk_forward_rows": int(len(walk_forward)),
        "walk_forward_confidence": _walk_forward_confidence_text(walk_forward),
        "walk_forward_csv": str(walk_forward_csv),
        "research_audit_rows": int(len(research_audit)),
        "research_audit_csv": str(research_audit_csv),
        "research_horizon_audit_rows": int(len(research_horizon_audit)),
        "research_horizon_audit_csv": str(research_horizon_audit_csv),
        "research_outcome_rows": int(len(research_outcomes)),
        "research_outcomes_csv": str(research_outcomes_csv),
        "strategy_family_audit_rows": int(len(strategy_family_audit)),
        "strategy_family_audit_csv": str(strategy_family_audit_csv),
        "ticker_playbook_audit_rows": int(len(ticker_playbook_audit)),
        "ticker_playbook_audit_csv": str(ticker_playbook_audit_csv),
        "rolling_ticker_playbook_audit_rows": int(len(rolling_ticker_playbook_audit)),
        "rolling_ticker_playbook_audit_csv": str(rolling_ticker_playbook_audit_csv),
        "market_regime": market_regime,
        "open_position_summary": open_position_summary,
        "trade_tracking": tracking_summary,
        "trade_tracker_csv": str(trade_tracker_csv),
        "research_audit_verdicts": research_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
        if not research_audit.empty
        else {},
        "research_horizon_audit_verdicts": research_horizon_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
        if not research_horizon_audit.empty
        else {},
        "strategy_family_audit_verdicts": strategy_family_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
        if not strategy_family_audit.empty
        else {},
        "ticker_playbook_audit_verdicts": ticker_playbook_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
        if not ticker_playbook_audit.empty
        else {},
        "rolling_ticker_playbook_audit_verdicts": rolling_ticker_playbook_audit.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
        if not rolling_ticker_playbook_audit.empty
        else {},
        "research_audit_min_outcomes": RESEARCH_AUDIT_MIN_OUTCOMES,
        "research_audit_min_unique_setups": RESEARCH_AUDIT_MIN_UNIQUE_SETUPS,
        "family_audit_min_train_setups": FAMILY_AUDIT_MIN_TRAIN_SETUPS,
        "family_audit_min_validation_setups": FAMILY_AUDIT_MIN_VALIDATION_SETUPS,
        "ticker_playbook_min_train_setups": TICKER_PLAYBOOK_MIN_TRAIN_SETUPS,
        "ticker_playbook_min_validation_setups": TICKER_PLAYBOOK_MIN_VALIDATION_SETUPS,
        "rolling_ticker_playbook_min_forward_tests": ROLLING_PLAYBOOK_MIN_FORWARD_TESTS,
        "rolling_ticker_playbook_min_forward_dates": ROLLING_PLAYBOOK_MIN_FORWARD_DATES,
        "rolling_ticker_playbook_min_profit_factor": ROLLING_PLAYBOOK_MIN_PROFIT_FACTOR,
        "reuse_walk_forward_raw": bool(args.reuse_walk_forward_raw),
        "reuse_walk_forward_outcomes": bool(args.reuse_walk_forward_outcomes),
        "reuse_research_outcomes": bool(args.reuse_research_outcomes),
        "walk_forward_samples": max(0, int(args.walk_forward_samples)),
        "walk_forward_horizons": _parse_walk_forward_horizons(args.walk_forward_horizons),
        "min_backtest_edge": float(args.min_backtest_edge),
        "min_backtest_signals": int(args.min_backtest_signals),
        "max_backtest_setups": int(args.max_backtest_setups),
        "min_workup_signals": int(args.min_workup_signals),
        "min_swing_score": float(args.candidate_min_score),
        "allow_low_sample": bool(args.allow_low_sample),
        "allow_earnings_risk": bool(args.allow_earnings_risk),
        "allow_volatile_ic": bool(args.allow_volatile_ic),
        "allow_flow_conflict": bool(args.allow_flow_conflict),
        "max_bid_ask_to_price_pct": float(args.max_bid_ask_to_price_pct),
        "max_bid_ask_to_width_pct": float(args.max_bid_ask_to_width_pct),
        "max_short_delta": float(args.max_short_delta),
        "min_underlying_price": float(args.min_underlying_price),
        "min_debit_spread_price": float(args.min_debit_spread_price),
        "min_whale_appearances": int(args.min_whale_appearances),
        "report": str(report_path),
        "candidate_csv": str(candidate_csv),
        "current_setups_csv": str(current_setups_csv),
        "max_conviction_csv": str(max_conviction_csv),
        "trade_workup_csv": str(trade_workup_csv),
        "actionable_csv": str(actionable_csv),
        "patterns_csv": str(patterns_csv),
        "quote_replay_csv": str(quote_replay_csv) if quote_replay_mode != "off" else "",
        "raw_report": str(raw_report),
        "raw_csv": str(raw_csv),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote: {report_path}")
    print(f"Wrote: {current_setups_csv}")
    print(f"Wrote: {max_conviction_csv}")
    print(f"Wrote: {trade_workup_csv}")
    print(f"Wrote: {actionable_csv}")
    print(f"Wrote: {patterns_csv}")
    print(f"Wrote: {walk_forward_csv}")
    print(f"Wrote: {research_audit_csv}")
    print(f"Wrote: {research_horizon_audit_csv}")
    print(f"Wrote: {research_outcomes_csv}")
    print(f"Wrote: {strategy_family_audit_csv}")
    print(f"Wrote: {ticker_playbook_audit_csv}")
    print(f"Wrote: {rolling_ticker_playbook_audit_csv}")
    if not bool(args.no_trade_tracking):
        print(f"Wrote: {trade_tracker_csv}")
    return {
        "report": report_path,
        "candidate_csv": candidate_csv,
        "current_setups_csv": current_setups_csv,
        "max_conviction_csv": max_conviction_csv,
        "trade_workup_csv": trade_workup_csv,
        "actionable_csv": actionable_csv,
        "patterns_csv": patterns_csv,
        "quote_replay_csv": quote_replay_csv,
        "walk_forward_csv": walk_forward_csv,
        "research_audit_csv": research_audit_csv,
        "research_horizon_audit_csv": research_horizon_audit_csv,
        "research_outcomes_csv": research_outcomes_csv,
        "strategy_family_audit_csv": strategy_family_audit_csv,
        "ticker_playbook_audit_csv": ticker_playbook_audit_csv,
        "rolling_ticker_playbook_audit_csv": rolling_ticker_playbook_audit_csv,
        "trade_tracker_csv": trade_tracker_csv,
        "metadata": metadata_path,
        "raw_report": raw_report,
        "raw_csv": raw_csv,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
