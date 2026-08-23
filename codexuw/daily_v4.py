from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .catalysts import earnings_crosses_expiry, earnings_event_date, load_catalyst_context
from .confirmations import apply_confirmation_evidence, build_confirmation_evidence
from .confidence_calibration import (
    CONSERVATIVE_CONFIDENCE_STATUS,
    DEFAULT_EDGE_HISTORY_PATH,
    apply_confidence_calibration,
    build_default_walk_forward_calibration,
    confidence_high_ready,
)
from .data import (
    aggregate_bot_flow,
    aggregate_dark_pool_flow,
    dte_from_expiry,
    infer_asof_date,
    load_chain_oi,
    load_hot_chains,
    load_stock_screener,
    read_csv_export,
    safe_float,
)
from .credit_policy import (
    CREDIT_POLICY_VERSION,
    MIN_IV_RANK,
    MIN_REALIZED_VOL,
    PROFIT_TAKE_PCT,
    assess_credit_spread,
    credit_spread_confidence,
)
from .debit_policy import assess_debit_spread, debit_spread_confidence
from .daily import infer_report_mode, latest_dated_folder, live_planning_validation_note
from .engine import (
    apply_catalyst_context,
    apply_confidence_components,
    apply_confirmation_framework,
    apply_data_quality_gate,
    apply_final_quality_guards,
    apply_high_conviction_decision_marks,
    apply_oi_carryover,
    apply_portfolio_context,
    apply_replay_edge_model,
    assign_trade_statuses,
    build_data_quality_status,
    detect_regime,
    generate_candidates,
    is_etf_row,
    live_validate_and_score,
    select_index_fallback_pool,
    select_ticker_pool,
)
from .edge_model import EDGE_HISTORY_NAMESPACE
from .integrity_v42 import apply_schwab_price_context
from .realized_vol import attach_realized_vol
from .fallback_income import apply_fallback_income_status, build_fallback_income_candidates
from .goal_shadow import write_goal_shadow_outputs
from .liquidity_shift import (
    FIXED_LIQUID_UNIVERSE,
    INDEX_FLOW_TICKERS,
    SECTOR_BENCHMARKS,
    apply_liquidity_shift_context,
    build_liquidity_shift_signals,
    expand_pool_with_top_flow,
)
from .loss_review import apply_loss_review, load_recent_loss_review, load_recommendation_ledgers
from .macro_gates import build_macro_event_gates
from .missed_opportunity import build_missed_opportunity_audit
from .occ import parse_occ_symbol
from .performance import load_live_outcome_performance, load_recent_performance
from .payoff_calibration import (
    PROBATIONARY_PAYOFF_STATUS,
    apply_payoff_calibration,
    build_default_payoff_calibration,
    build_snapshot_replay_summary,
    write_payoff_calibration_outputs,
)
from .pipeline_versions import PIPELINE_NAME_V4, PIPELINE_VERSION_V4, pipeline_version_record
from .portfolio import fetch_portfolio_context, unavailable_portfolio_context
from .portfolio_capacity import write_portfolio_capacity_outputs
from .provenance import build_input_provenance
from .sector_rotation import apply_sector_rotation_context, build_live_sector_rotation
from .strategy_registry import apply_strategy_registry_gate, build_strategy_registry, strategy_key_for_row
from .symbol_credit_calibration import (
    apply_symbol_credit_calibration,
    build_symbol_credit_calibration,
    write_symbol_credit_calibration_outputs,
)
from .target_model import business_days_remaining
from .validation import select_systematic_date_folders
from .walk_forward_credit_book import (
    DIRECTIONAL_CREDIT_EXECUTION_OI_STATES,
    DIRECTIONAL_CREDIT_MAX_DTE,
    DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO,
    DIRECTIONAL_CREDIT_MAX_PCT_WIDTH,
    DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH,
    DIRECTIONAL_CREDIT_MIN_DTE,
    DIRECTIONAL_CREDIT_MIN_PCT_WIDTH,
    apply_walk_forward_credit_model,
    build_walk_forward_credit_model,
    write_walk_forward_credit_outputs,
)


DEFAULT_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
RUN_MODE_V4 = "EOD swing target plan"

HARD_BLOCKER_TOKENS = {
    "chain_error",
    "live_unavailable",
    "no_realistic_spread",
    "no_usable_liquidity",
    "bid_ask_too_wide",
    "earnings_within_7d",
    "risk_cap",
    "duplicate_exposure_breach",
    "correlated_exposure_breach",
    "negative_replay_edge",
}

NON_HARD_PRICE_TOKENS = {
    "credit_below_min_16pct_width",
    "debit_bad_reward_risk_or_credit_below_min",
    "replay_guard_credit_below_validated_band",
    "replay_guard_credit_above_validated_band",
    "thin_replay_sample",
    "news_catalyst_caution",
    "news_unconfirmed",
    "marginal_liquidity",
    "wide_bid_ask",
    "oi_carryover_contrary",
    "portfolio_warning",
    "large_existing_equity_exposure",
    "strategy_specific_validation_required",
    "regime_transition",
    "market_regime_alignment",
}

EXECUTE_QUALITY_BLOCKER_TOKENS = {
    "manual-confirmation scout": "manual confirmation required",
    "do not mark execute": "prior construction explicitly blocks Execute",
    "final_quality_guard": "final quality guard did not clear",
    "decision_final_quality_guard": "final quality guard did not clear",
    "debit_bad_reward_risk_or_credit_below_min": "debit reward/risk guard did not clear",
    "debit_replay_guard_bad_structure": "debit replay guard rejected structure",
    "debit_ev_not_supported": "debit EV guard requires review",
    "no_flow_edge_alignment": "flow and edge are not aligned",
    "price_action_trend": "price-action trend confirmation failed",
    "decision_score_below_medium": "decision score below Execute threshold",
    "news_catalyst_caution": "news/catalyst caution requires review",
    "news_unconfirmed": "earnings/news evidence is unresolved",
    "oi_carryover_contrary": "exact-leg OI conflicts with the trade direction",
    "thin_replay_sample": "historical edge sample is too thin for Execute",
    "data_gate_missing_portfolio_state": "portfolio state is unavailable",
    "data_gate_news_unconfirmed": "ticker news/earnings evidence is unresolved",
    "strategy_specific_validation_required": "strategy-specific historical/live parity validation is required",
}

V4_TARGET_TICKET_COLUMNS = [
    "rank",
    "lane",
    "ticker",
    "trade legs",
    "expiry",
    "DTE",
    "next-session swing entry target",
    "current Schwab mid/natural reference",
    "maximum profit",
    "profit target",
    "max loss",
    "one-contract risk",
    "reward/risk",
    "suggested size",
    "swing hold window",
    "stop/invalidation",
    "gap-risk plan +/-1% open",
    "EOD trend evidence",
    "flow/OI evidence",
    "GEX context",
    "liquidity context",
    "portfolio context",
    "regime fit",
    "catalyst/earnings risk",
    "target price methodology",
    "OCO bracket order logic",
    "blocker before entry",
    "manual review instruction",
    "why this is worth reviewing tomorrow",
    "display status",
    "final disposition",
    "setup family",
    "setup family key",
    "safety calibration flags",
    "expected win rate",
    "win-rate basis",
    "confidence evidence",
    "per-ticket replay edge",
    "payoff evidence",
    "confidence rating",
    "expected average win",
    "expected average loss",
    "expected value",
    "implied profit factor",
]


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _base_dir_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "base_dir", ""):
        return Path(args.base_dir).expanduser().resolve()
    date = _parse_date(getattr(args, "date", ""))
    if date is None:
        raise SystemExit("--date or --base-dir is required")
    return Path(getattr(args, "root", DEFAULT_ROOT)).expanduser().resolve() / str(date)


def _default_out_dir(root: Path, asof: dt.date, mode: str, overlay_date: dt.date | None = None) -> Path:
    if mode == "validation":
        return root / "out" / f"codexdaily_v4_validation_{asof}"
    if mode == "overlay":
        return root / "out" / f"codexdaily_v4_overlay_{asof}_overlay_{overlay_date or asof}"
    return root / "out" / f"codexdaily_v4_{asof}"


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Trading desk root. Used with --date.")
    parser.add_argument("--date", default="", help="Dated UW folder date, e.g. 2026-05-20.")
    parser.add_argument("--base-dir", default="", help="Dated UW folder. Overrides --date.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to out/codexdaily_v4_YYYY-MM-DD.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Discovery cap. Default 0 scans every eligible source ticker.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Candidate cap. Default 0 keeps every constructed candidate before scoring.")
    parser.add_argument(
        "--max-final-trades",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility flag. Ignored by Codex Daily V4 because no-miss reporting must not hide "
            "valid Execute or Swing Target rows; risk caps and sizing control execution."
        ),
    )
    parser.add_argument(
        "--risk-budget",
        type=float,
        default=0.0,
        help=(
            "Optional aggregate risk-planning lens. Default 0 means no aggregate slate budget is applied; "
            "V4 still shows every valid Execute/Target and only enforces the required per-ticket safety caps."
        ),
    )
    parser.add_argument("--bot-max-rows", type=int, default=0)
    parser.add_argument(
        "--dark-pool-weight",
        type=float,
        default=0.0,
        help="Bounded equity dark-pool contribution (0..0.25) to combined directional flow; default 0 until replay-validated.",
    )
    parser.add_argument("--offline", action="store_true", help="Test-only: skip Schwab live chain validation.")
    parser.add_argument("--skip-portfolio", action="store_true", help="Skip Schwab portfolio pull; blocks live-quality Execute in V4.")
    parser.add_argument("--skip-catalysts", action="store_true", help="Skip local browser/news catalyst checks.")
    parser.add_argument("--skip-recent-performance", action="store_true")
    parser.add_argument("--schwab-snapshot-dir", default="", help="Existing Schwab chain snapshot directory for reproducible reruns.")
    parser.add_argument("--report-mode", default="post-close", choices=["auto", "pre-market", "post-close", "historical"])
    parser.add_argument("--max-risk-per-trade", type=float, default=0.0)
    parser.add_argument("--max-risk-per-day", type=float, default=0.0)
    parser.add_argument("--max-open-risk-by-ticker", type=float, default=0.0)
    parser.add_argument("--max-correlated-sector-exposure", type=float, default=0.0)
    parser.add_argument("--max-total-open-risk", type=float, default=0.0)
    parser.add_argument("--max-contracts-per-trade", type=float, default=20.0)
    parser.add_argument("--daily-loss-limit", type=float, default=0.0)
    parser.add_argument("--weekly-loss-limit", type=float, default=0.0)
    parser.add_argument("--monthly-loss-limit", type=float, default=0.0)
    parser.add_argument("--monthly-profit-target", type=float, default=10_000.0)
    parser.add_argument("--validation-account-value", type=float, default=0.0)
    parser.add_argument("--validation-risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--validation-max-active-ticker-share", type=float, default=0.20)
    parser.add_argument("--validation-max-active-sector-share", type=float, default=0.40)
    parser.add_argument("--month-to-date-realized-pnl", type=float, default=0.0)
    parser.add_argument("--open-unrealized-pnl", type=float, default=0.0)
    parser.add_argument("--max-monthly-drawdown", type=float, default=0.0)
    parser.add_argument("--minimum-expected-value-per-dollar-risk", type=float, default=0.01)
    parser.add_argument(
        "--risk-mandate",
        default="target-growth",
        choices=["capital-preservation", "balanced", "target-growth"],
    )
    parser.add_argument("--index-income-mode", default="primary", choices=["disabled", "fallback", "primary"])
    parser.add_argument(
        "--portfolio-income-mode",
        default="trading-sleeve-only",
        choices=["disabled", "trading-sleeve-only", "existing-core-review"],
    )
    parser.add_argument("--covered-income-allowed-tickers", default="")
    parser.add_argument("--loss-lookback-days", type=int, default=30)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(argv) if argv is not None else None
    if raw is None:
        import sys

        raw = list(sys.argv[1:])
    if raw and raw[0] in {"-h", "--help"}:
        pass
    elif not raw or raw[0].startswith("-"):
        raw = ["run", *raw]

    parser = argparse.ArgumentParser(
        description=(
            "Codex Daily V4 EOD professional swing-trading target pipeline. "
            "Primary product: Swing Target Tickets For Tomorrow."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Build the Codex Daily V4 EOD swing target plan.")
    _add_common_run_args(run)

    validate = sub.add_parser("validate", help="Run Codex Daily V4 validation over systematic recent source-complete dates.")
    _add_common_run_args(validate)
    validate.add_argument("--as-of", default="", help="Validation as-of date. Defaults to latest dated folder.")
    validate.add_argument("--latest-n", type=int, default=5)
    validate.add_argument("--run-live", action="store_true", help="Run the Schwab-scored core for selected dates before V4 comparison.")

    overlay = sub.add_parser("overlay", help="Build a Codex Daily V4 overlay report from a newer chain-oi file.")
    overlay.add_argument("--root", default=str(DEFAULT_ROOT))
    overlay.add_argument("--date", required=True, help="Original analysis date.")
    overlay.add_argument("--prior-out-dir", default="", help="Prior V4 or scored-core output directory.")
    overlay.add_argument("--overlay-file", required=True, help="chain-oi-changes overlay CSV or ZIP.")
    overlay.add_argument("--overlay-date", default="", help="Overlay file date if not inferable from filename.")
    overlay.add_argument("--out-dir", default="", help="Defaults to out/codexdaily_v4_overlay_DATE_overlay_OVERLAYDATE.")
    return parser.parse_args(raw)


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _unique_semicolon_context(*values: object) -> str:
    """Join evidence clauses without repeating merged portfolio/source notes."""
    parts: list[str] = []
    normalized: list[str] = []
    for value in values:
        for raw in _clean(value).split(";"):
            part = raw.strip()
            key = re.sub(r"\s+", " ", part).lower()
            if not key:
                continue
            if any(key == seen or (len(key) >= 12 and key in seen) for seen in normalized):
                continue
            parts.append(part)
            normalized.append(key)
    return "; ".join(parts)


def _v4_text(value: object) -> object:
    if isinstance(value, str):
        return (
            value.replace("Codex Daily V3", PIPELINE_NAME_V4)
            .replace("V3 confirmation", "V4 confirmation")
            .replace("v3 confirmation", "v4 confirmation")
            .replace("V3", "V4")
        )
    if isinstance(value, dict):
        return {key: _v4_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_v4_text(item) for item in value]
    return value


def _normalize_v4_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    out = df.copy()
    out = out.rename(
        columns={
            "v3_confirmation_status": "v4_confirmation_status",
            "v3_confirmation_reason": "v4_confirmation_reason",
        }
    )
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(_v4_text)
    return out


def _money_value(value: object) -> float:
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    return safe_float(value)


def _money(value: object, *, blank: str = "") -> str:
    number = _money_value(value)
    return f"${number:,.2f}" if math.isfinite(number) else blank


def _pct(value: object, *, blank: str = "") -> str:
    number = safe_float(value)
    return f"{number:.1%}" if math.isfinite(number) else blank


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_overlay_date_from_name(path: Path) -> dt.date | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()


def _load_overlay_chain_oi_file(path: Path, *, asof: dt.date) -> pd.DataFrame:
    df = read_csv_export(path)
    if "option_symbol" not in df.columns:
        return pd.DataFrame()
    parsed = df["option_symbol"].map(parse_occ_symbol)
    parsed_ticker = parsed.map(lambda x: x.root if x else "")
    underlying_ticker = (
        df["underlying_symbol"].fillna("").astype(str).str.strip().str.upper()
        if "underlying_symbol" in df.columns
        else pd.Series("", index=df.index, dtype=object)
    )
    df["ticker"] = parsed_ticker.where(parsed_ticker.astype(str).str.strip().ne(""), underlying_ticker)
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    for col in [
        "oi_diff_plain",
        "oi_change",
        "curr_oi",
        "last_oi",
        "volume",
        "last_fill",
        "last_bid",
        "last_ask",
        "prev_total_premium",
        "prev_neutral_volume",
        "prev_mid_volume",
        "prev_bid_volume",
        "prev_ask_volume",
        "prev_stock_multi_leg_volume",
        "prev_multi_leg_volume",
        "curr_vol",
        "prev_vol",
        "trades",
        "avg_price",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["source_path"] = str(path)
    return df


def _overlay_candidate_key(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("ticker") or ""),
            str(row.get("strategy") or row.get("direction") or ""),
            str(row.get("expiry") or ""),
            str(row.get("short_leg") or row.get("sell_leg") or ""),
            str(row.get("long_leg") or row.get("buy_leg") or ""),
        ]
    )


def _recommendation_rank(status: object) -> int:
    text = str(status or "")
    if "Execute" in text:
        return 5
    if "Swing Target" in text:
        return 4
    if "Scout" in text:
        return 3
    if "Research" in text:
        return 2
    if "Avoid" in text:
        return 1
    return 0


def _compare_v4_overlay_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before is None:
        before = pd.DataFrame()
    if after is None:
        after = pd.DataFrame()
    before = before.copy()
    after = after.copy()
    columns = [
        "candidate_key",
        "ticker",
        "trade",
        "previous_status",
        "new_status",
        "previous_oi_support",
        "new_oi_support_or_conflict",
        "recommendation_change",
        "removed_trade",
        "changed_live_pricing",
        "exact_reason",
    ]
    if before.empty and after.empty:
        return pd.DataFrame(columns=columns)
    before["_key"] = before.apply(_overlay_candidate_key, axis=1) if not before.empty else pd.Series(dtype=object)
    after["_key"] = after.apply(_overlay_candidate_key, axis=1) if not after.empty else pd.Series(dtype=object)
    bmap = before.set_index("_key", drop=False) if not before.empty else pd.DataFrame()
    amap = after.set_index("_key", drop=False) if not after.empty else pd.DataFrame()
    keys = sorted(set(bmap.index) | set(amap.index))
    rows: list[dict[str, Any]] = []
    for key in keys:
        if not bmap.empty and key in bmap.index:
            old_lookup = bmap.loc[key]
            old = old_lookup.iloc[0] if isinstance(old_lookup, pd.DataFrame) else old_lookup
        else:
            old = pd.Series(dtype=object)
        if not amap.empty and key in amap.index:
            new_lookup = amap.loc[key]
            new = new_lookup.iloc[0] if isinstance(new_lookup, pd.DataFrame) else new_lookup
        else:
            new = pd.Series(dtype=object)
        row = new if not new.empty else old
        old_status = str(old.get("trade_status") or old.get("v4_disposition") or "") if not old.empty else ""
        new_status = str(new.get("trade_status") or new.get("v4_disposition") or "") if not new.empty else ""
        old_oi = str(old.get("oi_carryover_status") or "") if not old.empty else ""
        new_oi = str(new.get("oi_carryover_status") or "") if not new.empty else ""
        old_reason = (
            str(old.get("v4_direct_disposition_reason") or old.get("trade_status_reason") or old.get("primary_blocker") or "")
            if not old.empty
            else ""
        )
        new_reason = (
            str(new.get("v4_direct_disposition_reason") or new.get("trade_status_reason") or new.get("primary_blocker") or "")
            if not new.empty
            else ""
        )
        old_mid = safe_float(old.get("mid_credit"), safe_float(old.get("mid_debit"))) if not old.empty else math.nan
        new_mid = safe_float(new.get("mid_credit"), safe_float(new.get("mid_debit"))) if not new.empty else math.nan
        pricing_refreshed = _clean(new.get("overlay_live_pricing_refreshed")).lower() in {"true", "1", "yes"}
        if math.isfinite(old_mid) and math.isfinite(new_mid):
            same_price = abs(old_mid - new_mid) < 0.005
            live_changed = (
                "refreshed_unchanged" if pricing_refreshed and same_price
                else "refreshed_changed" if pricing_refreshed
                else "not refreshed" if same_price
                else "changed"
            )
        else:
            live_changed = "not available"
        if old_status == new_status and old_oi == new_oi and live_changed == "not refreshed":
            continue
        rank_delta = _recommendation_rank(new_status) - _recommendation_rank(old_status)
        if rank_delta > 0:
            rec_change = "upgraded"
        elif rank_delta < 0:
            rec_change = "downgraded"
        elif old_status != new_status:
            rec_change = "changed"
        else:
            rec_change = "unchanged_status"
        rows.append(
            {
                "candidate_key": key,
                "ticker": _clean(row.get("ticker")).upper(),
                "trade": _clean(row.get("strategy") or row.get("direction")),
                "previous_status": old_status or "new",
                "new_status": new_status or "removed",
                "previous_oi_support": old_oi,
                "new_oi_support_or_conflict": new_oi,
                "recommendation_change": rec_change,
                "removed_trade": bool(new.empty),
                "changed_live_pricing": live_changed,
                "exact_reason": new_reason or old_reason or "OI overlay changed candidate context",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _read_prior_v4_scored(prior_out_dir: Path, asof: dt.date) -> pd.DataFrame:
    candidates = [
        prior_out_dir / f"codexdaily_v4_scored_reference_{asof}.csv",
        prior_out_dir / f"codexdaily_v4_raw_universe_{asof}.csv",
        prior_out_dir / f"codexdaily_v4_candidate_disposition_{asof}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No prior Codex Daily V4 scored CSV found in {prior_out_dir} for {asof}")


def _read_prior_v4_top_flow(prior_out_dir: Path, asof: dt.date) -> pd.DataFrame:
    raw = prior_out_dir / f"codexdaily_v4_raw_universe_{asof}.csv"
    if not raw.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw)
    if "top_flow_rank" not in df.columns:
        return pd.DataFrame()
    return df[df["top_flow_rank"].notna()].copy()


def _append_token(value: object, token: str) -> str:
    parts = [item.strip() for item in str(value or "").split(";") if item.strip() and item.strip().lower() != "nan"]
    if token and token not in parts:
        parts.append(token)
    return ";".join(parts)


def _append_note(value: object, note: str) -> str:
    existing = _clean(value)
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def _parse_ledger_date(value: object) -> dt.date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _add_trading_days(start: dt.date, days: int) -> dt.date:
    current = start
    added = 0
    while added < days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def _setup_family_key(row: pd.Series | dict[str, Any]) -> str:
    ticker = _clean(row.get("ticker") or row.get("Ticker")).upper()
    strategy = _clean(row.get("strategy") or row.get("Trade") or row.get("trade legs")).lower()
    expiry = _clean(row.get("expiry") or row.get("Expiry") or row.get("expiration_date"))
    if not ticker and not strategy and not expiry:
        return ""
    return f"{ticker} :: {strategy} :: {expiry}"


def _safety_research_reason(row: pd.Series | dict[str, Any]) -> str:
    penalties = _clean(row.get("penalties")).lower()
    note = _clean(row.get("v4_safety_notes"))
    if "v4_strategy_slump_muted" in penalties:
        return note or "V4 strategy-slump mute: recent family loss rate exceeded 40%"
    if "v4_negative_shadow_ev" in penalties and not _medium_debit_sleeve_eligible(row):
        return note or "V4 shadow backtest: recent family EV is negative"
    if "recent_loss_family:" in penalties:
        return note or "recent losing setup family requires Research disposition unless materially different"
    return ""


def _is_debit(row: pd.Series | dict[str, Any]) -> bool:
    entry_type = _clean(row.get("entry_type")).lower()
    kind = _clean(row.get("strategy_kind")).lower()
    if entry_type in {"credit", "debit"}:
        return entry_type == "debit"
    if kind in {"credit", "debit"}:
        return kind == "debit"
    text = f"{row.get('strategy', '')} {row.get('direction', '')}".lower()
    return "debit" in text or "bull call" in text or "bear put" in text


def _is_credit(row: pd.Series | dict[str, Any]) -> bool:
    entry_type = _clean(row.get("entry_type")).lower()
    kind = _clean(row.get("strategy_kind")).lower()
    if entry_type in {"credit", "debit"}:
        return entry_type == "credit"
    if kind in {"credit", "debit"}:
        return kind == "credit"
    return not _is_debit(row)


def _setup_family(row: pd.Series | dict[str, Any]) -> str:
    key = _clean(row.get("strategy_registry_key") or row.get("strategy_key"))
    if key and key not in {
        "bull_call_debit_vertical",
        "bear_put_debit_vertical",
        "bull_put_credit_vertical",
        "bear_call_credit_vertical",
    }:
        return "Complex / non-vertical"
    strategy = _clean(row.get("strategy") or row.get("trade legs") or row.get("Trade")).lower()
    if "cash-secured" in strategy or "wheel" in strategy:
        return "Wheel/Cash"
    if "debit" in strategy or "bull call" in strategy or "bear put" in strategy:
        return "Debit spreads"
    if "credit" in strategy or "bull put" in strategy or "bear call" in strategy:
        return "Credit spreads"
    return "Other"


def build_v4_safety_calibration(
    *,
    scored: pd.DataFrame,
    outcome_ledger: pd.DataFrame,
    asof: dt.date,
    lookback_days: int = 14,
) -> pd.DataFrame:
    columns = [
        "setup_family_key",
        "ticker",
        "strategy",
        "expiry",
        "window_start",
        "window_end",
        "outcome_count",
        "loss_count",
        "loss_rate",
        "expected_value",
        "strategy_slump_muted",
        "mute_days",
        "mute_until",
        "shadow_backtest_status",
        "source",
    ]
    scored_families: dict[str, dict[str, Any]] = {}
    if scored is not None and not scored.empty:
        for _, row in scored.iterrows():
            key = _setup_family_key(row)
            if not key:
                continue
            scored_families.setdefault(
                key,
                {
                    "setup_family_key": key,
                    "ticker": _clean(row.get("ticker")).upper(),
                    "strategy": _clean(row.get("strategy")),
                    "expiry": _clean(row.get("expiry")),
                },
            )

    if outcome_ledger is None or outcome_ledger.empty or "realized_pnl" not in outcome_ledger.columns:
        rows = []
        for key, meta in scored_families.items():
            row = dict(meta)
            row.update(
                {
                    "window_start": str(asof - dt.timedelta(days=lookback_days)),
                    "window_end": str(asof),
                    "outcome_count": 0,
                    "loss_count": 0,
                    "loss_rate": math.nan,
                    "expected_value": math.nan,
                    "strategy_slump_muted": False,
                    "mute_days": 0,
                    "mute_until": "",
                    "shadow_backtest_status": "insufficient_history",
                    "source": "no_recent_outcome_ledger",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    df = outcome_ledger.copy()
    date_col = next((col for col in ["report_date", "asof", "date", "entry_date", "exit_date"] if col in df.columns), "")
    if not date_col:
        return build_v4_safety_calibration(scored=scored, outcome_ledger=pd.DataFrame(), asof=asof, lookback_days=lookback_days)
    df["_date"] = df[date_col].map(_parse_ledger_date)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    df["_family_key"] = df.apply(_setup_family_key, axis=1)
    cutoff = asof - dt.timedelta(days=lookback_days)
    recent = df[
        df["_date"].notna()
        & (df["_date"] < asof)
        & (df["_date"] >= cutoff)
        & df["realized_pnl"].notna()
        & df["_family_key"].astype(str).str.len().gt(0)
    ].copy()

    rows: list[dict[str, Any]] = []
    for key, meta in scored_families.items():
        part = recent[recent["_family_key"].eq(key)].copy()
        if part.empty:
            row = dict(meta)
            row.update(
                {
                    "window_start": str(cutoff),
                    "window_end": str(asof),
                    "outcome_count": 0,
                    "loss_count": 0,
                    "loss_rate": math.nan,
                    "expected_value": math.nan,
                    "strategy_slump_muted": False,
                    "mute_days": 0,
                    "mute_until": "",
                    "shadow_backtest_status": "insufficient_history",
                    "source": "recent_outcome_ledger",
                }
            )
            rows.append(row)
            continue
        pnl = pd.to_numeric(part["realized_pnl"], errors="coerce").dropna()
        outcome_count = int(len(pnl))
        loss_count = int((pnl < 0).sum())
        loss_rate = loss_count / outcome_count if outcome_count else math.nan
        expected_value = float(pnl.mean()) if outcome_count else math.nan
        muted = math.isfinite(loss_rate) and loss_rate > 0.40
        if math.isfinite(expected_value) and expected_value < 0:
            shadow_status = "negative_ev"
        elif outcome_count:
            shadow_status = "non_negative_ev"
        else:
            shadow_status = "insufficient_history"
        row = dict(meta)
        row.update(
            {
                "window_start": str(cutoff),
                "window_end": str(asof),
                "outcome_count": outcome_count,
                "loss_count": loss_count,
                "loss_rate": round(loss_rate, 4) if math.isfinite(loss_rate) else math.nan,
                "expected_value": round(expected_value, 2) if math.isfinite(expected_value) else math.nan,
                "strategy_slump_muted": bool(muted),
                "mute_days": 5 if muted else 0,
                "mute_until": str(_add_trading_days(asof, 5)) if muted else "",
                "shadow_backtest_status": shadow_status,
                "source": _clean(part.iloc[0].get("ledger_source")) or "recent_outcome_ledger",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def apply_v4_safety_calibration(scored: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    for col in ["penalties", "trade_status_reason", "v4_safety_flags", "v4_safety_notes", "v4_setup_family_key"]:
        if col not in out.columns:
            out[col] = ""
    if calibration is None or calibration.empty:
        out["v4_setup_family_key"] = out.apply(_setup_family_key, axis=1)
        return out
    cal = calibration.set_index("setup_family_key", drop=False)
    muted = set(cal[cal["strategy_slump_muted"].astype(bool)].index)
    negative_ev = set(cal[cal["shadow_backtest_status"].astype(str).eq("negative_ev")].index)
    for idx, row in out.iterrows():
        key = _setup_family_key(row)
        out.at[idx, "v4_setup_family_key"] = key
        if not key or key not in cal.index:
            continue
        cal_row = cal.loc[key]
        if isinstance(cal_row, pd.DataFrame):
            cal_row = cal_row.iloc[0]
        flags = []
        notes = []
        if key in muted:
            flags.append("Strategy Slump Muted")
            notes.append(
                "loss rate "
                f"{safe_float(cal_row.get('loss_rate'), 0.0):.0%} over {int(safe_float(cal_row.get('outcome_count'), 0))} recent outcomes; "
                f"mute until {cal_row.get('mute_until')}"
            )
            out.at[idx, "penalties"] = _append_token(out.at[idx, "penalties"], "v4_strategy_slump_muted")
            if not _is_hard_blocked(row):
                out.at[idx, "trade_status"] = "Research"
                out.at[idx, "trade_tier"] = "Research"
        if _clean(row.get("trade_status")) == "Execute" and key in negative_ev:
            flags.append("Negative Shadow EV")
            notes.append(
                f"14-day setup-family EV ${safe_float(cal_row.get('expected_value'), 0.0):,.2f}; Execute downgraded to Research"
            )
            out.at[idx, "penalties"] = _append_token(out.at[idx, "penalties"], "v4_negative_shadow_ev")
            if not _is_hard_blocked(row):
                out.at[idx, "trade_status"] = "Research"
                out.at[idx, "trade_tier"] = "Research"
        if flags:
            out.at[idx, "v4_safety_flags"] = _append_note(out.at[idx, "v4_safety_flags"], ", ".join(flags))
            out.at[idx, "v4_safety_notes"] = _append_note(out.at[idx, "v4_safety_notes"], " | ".join(notes))
            out.at[idx, "trade_status_reason"] = _append_note(out.at[idx, "trade_status_reason"], out.at[idx, "v4_safety_notes"])
    return out


def _leg_label(symbol: object) -> str:
    parsed = parse_occ_symbol(symbol)
    if parsed is None:
        return _clean(symbol)
    return f"{parsed.root} {parsed.expiry} {parsed.strike:g}{parsed.right}"


def _trade_legs(row: pd.Series | dict[str, Any]) -> str:
    existing = _clean(row.get("Trade") or row.get("trade legs"))
    if existing and not parse_occ_symbol(existing):
        return existing
    strategy = _clean(row.get("strategy") or row.get("direction")) or "Option structure"
    legs_text = _clean(row.get("legs_json"))
    if legs_text:
        try:
            legs = json.loads(legs_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            legs = []
        rendered = []
        for leg in legs if isinstance(legs, list) else []:
            instrument = _clean(leg.get("instrument")).lower()
            quantity = int(safe_float(leg.get("quantity"), 0.0))
            if instrument == "stock":
                rendered.append(f"hold {abs(quantity)} shares")
                continue
            side = "buy" if quantity > 0 else "sell"
            quantity_text = f" {abs(quantity)}x" if abs(quantity) != 1 else ""
            rendered.append(f"{side}{quantity_text} {_leg_label(leg.get('symbol'))}")
        if rendered:
            return f"{strategy}: " + " / ".join(rendered)
    short = _clean(row.get("short_leg") or row.get("sell_leg") or row.get("short_leg_eod"))
    long = _clean(row.get("long_leg") or row.get("buy_leg") or row.get("long_leg_eod"))
    if short and long:
        short_label = _leg_label(short)
        long_label = _leg_label(long)
        if _is_debit(row):
            return f"{strategy}: buy {long_label} / sell {short_label}"
        return f"{strategy}: sell {short_label} / buy {long_label}"
    if short:
        return f"{strategy}: sell {_leg_label(short)}"
    if long:
        return f"{strategy}: buy {_leg_label(long)}"
    return strategy


def _entry_target(row: pd.Series | dict[str, Any]) -> str:
    existing = _clean(row.get("Entry limit") or row.get("next-session swing entry target"))
    if existing:
        return existing
    required = safe_float(row.get("required_entry"), safe_float(row.get("target_entry")))
    if not math.isfinite(required):
        required = safe_float(row.get("fallback_target_credit"))
    if not math.isfinite(required):
        required = safe_float(row.get("credit") if _is_credit(row) else row.get("debit"))
    if not math.isfinite(required):
        return "fresh Schwab recheck"
    return f">= ${required:.2f} credit" if _is_credit(row) else f"<= ${required:.2f} debit"


def _mid_natural(row: pd.Series | dict[str, Any]) -> str:
    existing = _clean(row.get("Live mid/natural") or row.get("current Schwab mid/natural reference"))
    if existing:
        return existing
    if _is_debit(row):
        mid = safe_float(row.get("mid_debit"), safe_float(row.get("debit")))
        natural = safe_float(row.get("natural_debit"), safe_float(row.get("debit")))
    else:
        mid = safe_float(row.get("mid_credit"), safe_float(row.get("credit")))
        natural = safe_float(row.get("natural_credit"), safe_float(row.get("credit")))
    if math.isfinite(mid) and math.isfinite(natural):
        return f"{mid:.2f} / {natural:.2f}"
    if math.isfinite(mid):
        return f"{mid:.2f} / recheck"
    return "fresh Schwab recheck"


def _target_profit_value(row: pd.Series | dict[str, Any]) -> float:
    existing = _money_value(row.get("Target profit") or row.get("profit target"))
    if math.isfinite(existing):
        return existing
    explicit = safe_float(row.get("target_profit_total"))
    if math.isfinite(explicit):
        return explicit
    max_profit = safe_float(row.get("max_profit"))
    if not math.isfinite(max_profit):
        max_profit = _money_value(row.get("Max profit"))
    if not math.isfinite(max_profit):
        return math.nan
    return max_profit * (0.45 if _is_debit(row) else PROFIT_TAKE_PCT)


def _max_loss_value(row: pd.Series | dict[str, Any]) -> float:
    value = safe_float(row.get("max_loss"))
    if math.isfinite(value):
        return value
    return _money_value(row.get("Max loss") or row.get("max loss"))


def _maximum_profit_at_entry_value(
    row: pd.Series | dict[str, Any],
    disposition: str,
) -> float:
    """Return nominal maximum profit for one spread at the displayed entry limit."""
    target_text = (
        _ticket_entry_target(row, disposition)
        if _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row)
        else _entry_target(row)
    )
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", target_text)
    entry = safe_float(match.group(1)) if match else math.nan
    width = safe_float(row.get("spread_width"), safe_float(row.get("width")))
    if math.isfinite(entry) and entry > 0 and math.isfinite(width) and width > entry:
        return entry * 100.0 if _is_credit(row) else (width - entry) * 100.0
    explicit = safe_float(row.get("max_profit"))
    if math.isfinite(explicit):
        return explicit
    return _money_value(row.get("Max profit") or row.get("maximum profit"))


def _hard_blocker_reason(row: pd.Series | dict[str, Any]) -> str:
    hard = _clean(row.get("hard_rejects"))
    family_evidence = _validated_family_evidence(row) or _probationary_execution_ready(row)
    edge_sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size")))
    if hard:
        hard_parts = [part.strip() for part in re.split(r"[;,]", hard) if part.strip()]
        if family_evidence and math.isfinite(edge_sample) and edge_sample < V4_MIN_REPLAY_SAMPLE:
            hard_parts = [
                part
                for part in hard_parts
                if "negative_replay_edge" not in part.lower() and "thin_replay_sample" not in part.lower()
            ]
        if hard_parts:
            return ";".join(hard_parts)
    penalties = _clean(row.get("penalties")).lower()
    asof = _parse_date(_clean(row.get("v4_asof") or row.get("asof")))
    event_date = earnings_event_date(row)
    if earnings_crosses_expiry(row, asof=asof):
        return f"earnings/event {event_date} occurs on or before expiry"
    earnings_days = safe_float(row.get("catalyst_earnings_days"))
    catalyst_status = _clean(row.get("catalyst_status")).lower()
    if math.isfinite(earnings_days) and 0 <= earnings_days <= 7 and (
        "earnings_news_risk" in penalties or catalyst_status in {"caution", "mixed", "blocked"}
    ):
        return "earnings/event risk invalidates the structure"
    live_status = _clean(row.get("live_status"))
    if live_status and live_status not in {"PASS", "pass"}:
        return live_status
    natural = safe_float(row.get("natural_debit") if _is_debit(row) else row.get("natural_credit"))
    spread_width = safe_float(row.get("spread_width"))
    if math.isfinite(natural) and natural <= 0:
        return "invalid non-positive combination natural price"
    if math.isfinite(natural) and math.isfinite(spread_width) and spread_width > 0 and natural >= spread_width:
        return "invalid combination natural price at or above spread width"
    combination_width = _combination_quote_width_ratio(row)
    if math.isfinite(combination_width) and combination_width > 0.65:
        return "extreme combination bid/ask width"
    quote_width = safe_float(row.get("quote_width_pct"))
    if math.isfinite(quote_width) and quote_width > 0.65:
        return "extreme bid/ask width"
    avg = safe_float(row.get("edge_avg_pnl"))
    if _is_credit(row) and math.isfinite(avg) and avg <= 0 and not family_evidence:
        return "negative_edge_avg_pnl"
    replay = _clean(row.get("replay_ev_verdict")).lower()
    edge = _clean(row.get("edge_verdict")).lower()
    if replay.startswith("negative") and edge not in {"acceptable", "positive", "thin_sample"} and not family_evidence:
        return "negative edge with no offsetting thesis"
    for token in HARD_BLOCKER_TOKENS:
        if token in penalties:
            if (
                token == "negative_replay_edge"
                and family_evidence
                and math.isfinite(edge_sample)
                and edge_sample < V4_MIN_REPLAY_SAMPLE
            ):
                continue
            return token
    return ""


def _is_hard_blocked(row: pd.Series | dict[str, Any]) -> bool:
    return bool(_hard_blocker_reason(row))


def _truthy_flag(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _min_leg_session_volume(row: pd.Series | dict[str, Any]) -> float:
    explicit = safe_float(row.get("min_leg_volume"))
    if math.isfinite(explicit):
        return explicit
    legs = [safe_float(row.get("short_volume")), safe_float(row.get("long_volume"))]
    finite = [value for value in legs if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _execution_quote_blocker(row: pd.Series | dict[str, Any]) -> str:
    """Immediate orders need displayed size; post-close plans need a traded regular-session market."""
    market_open = _truthy_flag(row.get("market_session_open_at_validation"))
    if market_open:
        displayed_entry_size = safe_float(row.get("displayed_entry_size"))
        if not math.isfinite(displayed_entry_size) or displayed_entry_size < 1:
            return "natural-side option market has no displayed entry size"
        return ""
    if not _truthy_flag(row.get("regular_session_quote")):
        return "option quote was captured outside the regular options session"
    if "market_session_open_at_validation" in row:
        traded = _min_leg_session_volume(row)
        if not math.isfinite(traded) or traded < 1:
            return "no leg traded in the quoted session; next-session limit is not trustworthy"
        return ""
    displayed_entry_size = safe_float(row.get("displayed_entry_size"))
    if not math.isfinite(displayed_entry_size) or displayed_entry_size < 1:
        return "natural-side option market has no displayed entry size"
    return ""


def _execution_catalyst_verification_blocker(row: pd.Series | dict[str, Any]) -> str:
    """Keep unverified single-name news visible without suppressing setup discovery."""
    if is_etf_row(pd.Series(row)):
        return ""
    status = _clean(row.get("catalyst_status")).lower()
    news_quality = _clean(row.get("catalyst_news_quality")).lower()
    if status not in {"", "unknown"} and "news unconfirmed" not in news_quality:
        return ""
    event_date = earnings_event_date(row)
    if event_date is None:
        return "earnings date unresolved; ticker-specific news/catalyst status is unconfirmed; same-session verification required"
    return (
        "ticker-specific news/catalyst status is unconfirmed; "
        f"next structured earnings date is {event_date}; same-session verification required"
    )


def _execute_quality_blocker(
    row: pd.Series | dict[str, Any],
    *,
    include_entry_price: bool = True,
    include_execution_quote: bool = True,
    include_catalyst_verification: bool = True,
) -> str:
    text = ";".join(
        [
            _clean(row.get("trade_status_reason")),
            _clean(row.get("penalties")),
            _clean(row.get("primary_blocker")),
            _clean(row.get("what_must_improve")),
            _clean(row.get("data_quality_blockers")),
        ]
    ).lower()
    probationary_evidence = _probationary_execution_ready(row)
    medium_debit_evidence = _medium_debit_sleeve_eligible(row)
    walk_forward_credit_evidence = _walk_forward_credit_execution_ready(row)
    directional_credit_evidence = _directional_credit_execution_ready(row)
    family_evidence = (
        _validated_family_evidence(row)
        or probationary_evidence
        or medium_debit_evidence
        or walk_forward_credit_evidence
    )
    if include_execution_quote:
        quote_blocker = _execution_quote_blocker(row)
        if quote_blocker:
            return quote_blocker
    if include_catalyst_verification:
        catalyst_blocker = _execution_catalyst_verification_blocker(row)
        if catalyst_blocker:
            return catalyst_blocker
    if _probationary_payoff_ready(row) and not probationary_evidence:
        return "probationary route is observation-only until post-activation outcomes mature"
    calibration_status = _clean(row.get("confidence_calibration_status")).upper()
    if _is_debit(row) and calibration_status != "PASS" and not medium_debit_evidence:
        return "debit family walk-forward calibration is not validated"
    if not _payoff_evidence_ready(row) and not medium_debit_evidence:
        return "realized payoff lane did not pass cost-stressed walk-forward validation"
    primary_blocker = _clean(row.get("primary_blocker")).lower()
    for token, reason in EXECUTE_QUALITY_BLOCKER_TOKENS.items():
        if token in text:
            if medium_debit_evidence and token in {
                "final_quality_guard",
                "decision_final_quality_guard",
                "debit_bad_reward_risk_or_credit_below_min",
                "debit_replay_guard_bad_structure",
                "debit_ev_not_supported",
                "no_flow_edge_alignment",
                "price_action_trend",
                "decision_score_below_medium",
                "thin_replay_sample",
                "strategy_specific_validation_required",
            }:
                # The frozen debit model already consumes structure economics,
                # regime, flow, OI, technicals, IV/HV, and liquidity. Reusing
                # these legacy proxy blockers would make that validated route
                # unreachable. Independent live-risk gates remain mandatory.
                continue
            if (_symbol_credit_execution_ready(row) or walk_forward_credit_evidence) and token in {
                "price_action_trend",
                "decision_score_below_medium",
            }:
                # The validated credit map is intentionally contrarian. The
                # generic confirmation layer is trend-following, so applying
                # both makes the route logically impossible even after its
                # own walk-forward, live-policy, OI, and pricing gates pass.
                continue
            if family_evidence and token == "thin_replay_sample":
                continue
            if (
                family_evidence
                and token in {"final_quality_guard", "decision_final_quality_guard"}
                and primary_blocker.startswith("thin_replay_sample")
            ):
                continue
            return reason
    if _clean(row.get("catalyst_status")).lower() == "caution":
        return "news/catalyst caution requires review"
    if _clean(row.get("oi_carryover_status")).lower() == "contrary":
        return "exact-leg OI conflicts with the trade direction"
    if _is_debit(row):
        debit_ok, debit_reasons = assess_debit_spread(row, live=True)
        if not debit_ok and medium_debit_evidence:
            model_covered_prefixes = {
                "side_aware_bot_or_directional_contract_flow_required",
                "regime_not_aligned",
                "dte_outside_7_10_or_22_45",
                "debit_pct_width_above_0.45",
                "reward_risk_below_1.25",
                "breakeven_expected_move_ratio_below_0.50",
                "flow_alignment_below_0.20",
                "exact_leg_oi_not_confirmed",
                "debit_edge_sample_below_12",
                "debit_edge_pf_below_1.25",
                "debit_edge_avg_pnl_not_positive",
            }
            debit_reasons = [
                reason
                for reason in debit_reasons
                if _clean(reason).split(":", 1)[0] not in model_covered_prefixes
                and not _clean(reason).startswith("contract_flow_")
            ]
            debit_ok = not debit_reasons
        if not debit_ok:
            return "debit quality policy: " + "; ".join(debit_reasons)
    if _is_credit(row):
        if directional_credit_evidence:
            # The directional lane has its own frozen DTE, credit-band,
            # expected-move, flow, OI, and quote-width contract. Reapplying
            # the unrelated generic credit map here used narrower bands and
            # made a validated route logically unreachable.
            credit_ok, credit_reasons = True, []
        else:
            credit_row = dict(row)
            if _symbol_credit_execution_ready(row):
                credit_row["regime_trend"] = row.get("symbol_regime_trend")
                credit_row["regime"] = row.get("symbol_regime_trend")
                credit_row["edge_sample_size"] = row.get("symbol_credit_sample_size")
                credit_row["edge_profit_factor"] = row.get("symbol_credit_stress_profit_factor_10pct")
                credit_row["edge_avg_pnl"] = row.get("symbol_credit_stress_average_pnl_10pct")
            elif walk_forward_credit_evidence:
                credit_row["edge_sample_size"] = row.get("walk_forward_credit_oos_sample_size")
                credit_row["edge_profit_factor"] = row.get("walk_forward_credit_stress_profit_factor_10pct")
                credit_row["edge_avg_pnl"] = row.get("walk_forward_credit_stress_average_pnl_10pct")
            credit_ok, credit_reasons = assess_credit_spread(credit_row, live=True)
        if not credit_ok and family_evidence:
            route_key = _clean(row.get("payoff_route_key") or row.get("payoff_group_key")).lower()
            route_level = _clean(row.get("payoff_route_level")).lower()
            flow_quality = _clean(row.get("flow_quality")).lower()
            evidence_backed_prefixes = {
                "credit_regime_not_aligned",
                "credit_edge_sample_below_12",
                "credit_edge_pf_below_1.25",
                "credit_edge_avg_pnl_not_positive",
            }
            if walk_forward_credit_evidence:
                # The v4.19 family model already consumes flow magnitude,
                # realized volatility and IV/HV, then requires aligned flow,
                # supportive OI and positive recent direction/regime health.
                # Reapplying the generic cutoffs here silently nullifies the
                # frozen walk-forward route that selected this one-lot row.
                evidence_backed_prefixes.update(
                    {
                        "flow_alignment_below_0.10",
                        "realized_vol_below_0.15",
                        "iv_hv_ratio_below_0.90",
                    }
                )
            if probationary_evidence:
                iv_rank = safe_float(row.get("iv_rank"))
                realized = safe_float(row.get("realized_volatility_30d"))
                if (
                    math.isfinite(iv_rank)
                    and iv_rank >= MIN_IV_RANK
                    and math.isfinite(realized)
                    and realized >= MIN_REALIZED_VOL
                ):
                    evidence_backed_prefixes.add("iv_hv_ratio_below_0.90")
            remaining_reasons: list[str] = []
            for reason in credit_reasons:
                prefix = _clean(reason).split(":", 1)[0]
                if prefix in evidence_backed_prefixes:
                    continue
                if (
                    prefix == "credit_pct_width_outside_0.25_0.30"
                    and route_level == "flow_cost"
                    and "cost=18to30" in route_key
                ):
                    continue
                remaining_reasons.append(reason)
            credit_reasons = remaining_reasons
            credit_ok = not credit_reasons
        if not credit_ok:
            return "credit quality policy: " + "; ".join(credit_reasons)
    bot_source = _clean(row.get("bot_flow_source_status")).lower()
    if bot_source.startswith("missing_bot_eod") and _clean(row.get("flow_quality")).lower() != "directional":
        return "aggregate bot option flow unavailable and contract flow is not independently directional"
    sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size")))
    if math.isfinite(sample) and sample < V4_MIN_REPLAY_SAMPLE and not family_evidence:
        return f"historical edge sample too thin for Execute (n={int(sample)})"
    match_level = _clean(row.get("edge_match_level")).lower()
    medium_debit_sleeve = _medium_debit_sleeve_eligible(row)
    if match_level not in V4_EXECUTE_EDGE_MATCH_LEVELS and not medium_debit_sleeve and not family_evidence:
        return f"pooled historical edge ({match_level or 'unavailable'}) cannot authorize Execute"
    edge_pf = safe_float(row.get("edge_profit_factor"))
    if not family_evidence and (not math.isfinite(edge_pf) or edge_pf < V4_EXECUTE_MIN_PROFIT_FACTOR):
        return (
            f"historical edge PF is {edge_pf:.2f}; must be at least {V4_EXECUTE_MIN_PROFIT_FACTOR:.2f}"
            if math.isfinite(edge_pf)
            else "historical edge PF is unavailable"
        )
    trend = _clean(row.get("regime_trend") or row.get("regime") or row.get("trend")).lower()
    if trend == "range" and not family_evidence and (
        sample < V4_RANGE_EXECUTE_MIN_SAMPLE
        or edge_pf < V4_RANGE_EXECUTE_MIN_PROFIT_FACTOR
        or match_level not in {"exact", "ticker_direction", "strategy_regime"}
    ):
        return (
            "range regime is not validated for Execute: requires specific historical evidence, "
            f"n>={V4_RANGE_EXECUTE_MIN_SAMPLE}, PF>={V4_RANGE_EXECUTE_MIN_PROFIT_FACTOR:.2f}"
        )
    expectancy_blocker = _post_pricing_expectancy_blocker(row)
    if expectancy_blocker:
        return expectancy_blocker
    if include_entry_price:
        safe_price_blocker = _expectancy_safe_price_blocker(row)
        if safe_price_blocker:
            return safe_price_blocker
    return ""


V4_EXPECTANCY_POLICY_VERSION = "v4.9-structure-aware-payoff-20260720"
V4_EXECUTE_MIN_PROFIT_FACTOR = 1.25
V4_MIN_REPLAY_SAMPLE = 12
V4_RANGE_EXECUTE_MIN_SAMPLE = 20
V4_RANGE_EXECUTE_MIN_PROFIT_FACTOR = 1.50
V4_EXECUTE_EDGE_MATCH_LEVELS = {"exact", "ticker_direction", "strategy_regime"}
V4_MEDIUM_DEBIT_EDGE_MATCH_LEVEL = "debit_policy_sleeve"
V4_FAMILY_EVIDENCE_MIN_SAMPLE = 100
V4_FAMILY_EVIDENCE_MIN_LOWER_BOUND = 0.60
V4_PROBATIONARY_MAX_EXECUTES = 1
V4_MEDIUM_DEBIT_MAX_EXECUTES = 1
V4_MEDIUM_DEBIT_MIN_REWARD_RISK = 1.25
V4_MEDIUM_DEBIT_MAX_QUOTE_WIDTH = 0.25
V4_MEDIUM_DEBIT_ENTRY_STRESS_PCT = 0.10
V4_MEDIUM_DEBIT_PROFIT_TAKE_PCT = 1.00


def _validated_family_evidence(row: pd.Series | dict[str, Any]) -> bool:
    """Use family evidence only when walk-forward calibration is statistically usable.

    This is the hierarchical fallback for sparse exact/ticker matches. It does not
    bypass pricing, flow/OI, earnings, liquidity, regime, portfolio, or expectancy
    checks. Failed debit calibration therefore remains non-executable.
    """
    status = _clean(row.get("confidence_calibration_status")).upper()
    tier = _clean(row.get("confidence_model_tier")).lower()
    sample = safe_float(row.get("confidence_calibration_sample_size"))
    lower_bound = safe_float(row.get("confidence_probability_lower_bound"))
    brier = safe_float(row.get("confidence_calibration_brier"))
    baseline = safe_float(row.get("confidence_calibration_baseline_brier"), 0.25)
    return bool(
        status == "PASS"
        and tier.endswith("_validated")
        and math.isfinite(sample)
        and sample >= V4_FAMILY_EVIDENCE_MIN_SAMPLE
        and math.isfinite(lower_bound)
        and lower_bound >= V4_FAMILY_EVIDENCE_MIN_LOWER_BOUND
        and math.isfinite(brier)
        and math.isfinite(baseline)
        and brier < baseline
        and _payoff_model_ready(row)
    )


def _probationary_payoff_ready(row: pd.Series | dict[str, Any]) -> bool:
    if not _is_credit(row):
        return False
    minimum_sample = safe_float(row.get("payoff_minimum_sample_required"), 20.0)
    direction = _clean(row.get("direction"))
    trend = _clean(row.get("regime_trend") or row.get("regime") or row.get("trend")).lower()
    route_key = _clean(row.get("payoff_route_key") or row.get("payoff_group_key"))
    return bool(
        f"Credit|{direction}|{trend}".lower() in route_key.lower()
        and _clean(row.get("payoff_calibration_status")).upper() == PROBATIONARY_PAYOFF_STATUS
        and safe_float(row.get("payoff_sample_size"), 0.0) >= minimum_sample
        and safe_float(row.get("payoff_stress_10_average_pnl"), 0.0) > 0
        and safe_float(row.get("payoff_stress_10_profit_factor"), 0.0) >= V4_EXECUTE_MIN_PROFIT_FACTOR
        and safe_float(row.get("payoff_walk_forward_oos_sample"), 0.0) >= 5
        and safe_float(row.get("payoff_walk_forward_oos_average_pnl"), 0.0) > 0
        and safe_float(row.get("payoff_walk_forward_oos_profit_factor"), 0.0) >= V4_EXECUTE_MIN_PROFIT_FACTOR
        and safe_float(row.get("payoff_post_activation_oos_sample"), 0.0) < 2
    )


def _probationary_confidence_ready(row: pd.Series | dict[str, Any]) -> bool:
    status = _clean(row.get("confidence_calibration_status")).upper()
    sample = safe_float(row.get("confidence_calibration_sample_size"))
    lower_bound = safe_float(row.get("confidence_probability_lower_bound"))
    brier = safe_float(row.get("confidence_calibration_brier"))
    baseline = safe_float(row.get("confidence_calibration_baseline_brier"), 0.25)
    return bool(
        status in {"PASS", CONSERVATIVE_CONFIDENCE_STATUS}
        and math.isfinite(sample)
        and sample >= V4_FAMILY_EVIDENCE_MIN_SAMPLE
        and math.isfinite(lower_bound)
        and lower_bound >= V4_FAMILY_EVIDENCE_MIN_LOWER_BOUND
        and math.isfinite(brier)
        and math.isfinite(baseline)
        and brier < baseline
    )


def _symbol_credit_execution_ready(row: pd.Series | dict[str, Any]) -> bool:
    def profit_factor_ready(field: str) -> bool:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            return False
        return not math.isnan(value) and value >= V4_EXECUTE_MIN_PROFIT_FACTOR

    return bool(
        _is_credit(row)
        and _clean(row.get("symbol_credit_calibration_status")).upper() == "PASS"
        and _clean(row.get("symbol_credit_policy_pass")).lower() in {"true", "1", "yes"}
        and safe_float(row.get("symbol_credit_sample_size"), 0.0) >= 12
        and profit_factor_ready("symbol_credit_stress_profit_factor_10pct")
        and safe_float(row.get("symbol_credit_stress_average_pnl_10pct"), 0.0) > 0
        and safe_float(row.get("symbol_credit_oos_sample_size"), 0.0) >= 5
        and profit_factor_ready("symbol_credit_oos_profit_factor_10pct")
        and safe_float(row.get("symbol_credit_oos_average_pnl_10pct"), 0.0) > 0
        and safe_float(row.get("symbol_credit_postactivation_sample_size"), 0.0) >= 5
        and safe_float(row.get("symbol_credit_independent_episode_count"), 0.0) >= 5
        and safe_float(row.get("symbol_credit_wilson_lower_bound"), 0.0) >= 0.65
        and _clean(row.get("oi_carryover_status")).lower() in {"supportive", "matched_unconfirmed"}
    )


def _walk_forward_credit_execution_ready(row: pd.Series | dict[str, Any]) -> bool:
    strict_model_ready = bool(
        _is_credit(row)
        and _clean(row.get("walk_forward_credit_calibration_status")).upper() == "PASS"
        and _clean(row.get("walk_forward_credit_model_tier")).lower() == "medium"
        and _clean(row.get("walk_forward_credit_policy_pass")).lower() in {"true", "1", "yes"}
        and safe_float(row.get("walk_forward_credit_prediction"), -math.inf) >= 0.01
        and safe_float(row.get("walk_forward_credit_win_probability"), 0.0) >= 0.65
        and safe_float(row.get("walk_forward_credit_oos_sample_size"), 0.0) >= 20
        and safe_float(row.get("walk_forward_credit_wilson_lower_bound"), 0.0) >= 0.65
        and safe_float(row.get("walk_forward_credit_stress_profit_factor_10pct"), 0.0) >= 2.00
        and safe_float(row.get("walk_forward_credit_stress_average_pnl_10pct"), 0.0) > 0
        and safe_float(row.get("walk_forward_credit_positive_months"), 0.0) >= 4
        and _clean(row.get("oi_carryover_status")).lower() in {"supportive", "matched_unconfirmed"}
    )
    return strict_model_ready or _directional_credit_execution_ready(row)


def _directional_credit_execution_ready(row: pd.Series | dict[str, Any]) -> bool:
    direction = _clean(row.get("direction"))
    flow = safe_float(row.get("combined_flow_bias"), 0.0)
    flow_aligned = (direction == "Bull Put" and flow > 0) or (direction == "Bear Call" and flow < 0)
    credit_pct = safe_float(row.get("credit_pct_width"), safe_float(row.get("entry_credit_pct_width")))
    quote_width = safe_float(row.get("quote_width_pct"), safe_float(row.get("entry_quote_width_pct")))
    expected_move_ratio = safe_float(row.get("expected_move_ratio"))
    dte = safe_float(row.get("dte"), safe_float(row.get("entry_dte")))
    width = safe_float(row.get("spread_width"), safe_float(row.get("width")))
    natural_credit = safe_float(row.get("natural_credit"))
    execution_validation_status = _clean(
        row.get(
            "directional_credit_execution_validation_status",
            row.get("directional_credit_calibration_status"),
        )
    ).upper()
    family_validation_status = _clean(
        row.get("directional_credit_family_validation_status")
    ).upper()
    return bool(
        _is_credit(row)
        and direction in {"Bull Put", "Bear Call"}
        and flow_aligned
        and _clean(row.get("directional_credit_calibration_status")).upper() == "PASS"
        and execution_validation_status == "PASS"
        and family_validation_status in {"PASS", "PROBATIONARY"}
        and _clean(row.get("directional_credit_qualified")).lower() in {"true", "1", "yes"}
        and safe_float(row.get("directional_credit_execution_sample_size"), 0.0) >= 50
        and safe_float(row.get("directional_credit_execution_wilson_lower_bound"), 0.0) >= 0.78
        and safe_float(row.get("directional_credit_execution_stress_profit_factor_10pct"), 0.0) >= 2.00
        and safe_float(row.get("directional_credit_execution_stress_average_pnl_10pct"), 0.0) > 0
        and safe_float(row.get("directional_credit_execution_positive_months"), 0.0) >= 7
        and safe_float(row.get("directional_credit_family_sample_size"), 0.0) >= 15
        and safe_float(row.get("directional_credit_family_wilson_lower_bound"), 0.0) >= 0.65
        and safe_float(
            row.get("directional_credit_family_stress_profit_factor_10pct"), 0.0
        )
        >= 1.50
        and safe_float(row.get("directional_credit_execution_holdout_sample_size"), 0.0) >= 15
        and safe_float(row.get("directional_credit_execution_holdout_win_rate"), 0.0) >= 0.75
        and safe_float(
            row.get("directional_credit_execution_holdout_stress_profit_factor_10pct"), 0.0
        )
        >= 1.50
        and _clean(row.get("flow_quality")).lower() == "directional"
        and _clean(row.get("oi_carryover_status")).lower() in DIRECTIONAL_CREDIT_EXECUTION_OI_STATES
        and math.isfinite(credit_pct)
        and DIRECTIONAL_CREDIT_MIN_PCT_WIDTH <= credit_pct <= DIRECTIONAL_CREDIT_MAX_PCT_WIDTH
        and math.isfinite(quote_width)
        and quote_width <= DIRECTIONAL_CREDIT_MAX_QUOTE_WIDTH
        and math.isfinite(expected_move_ratio)
        and expected_move_ratio <= DIRECTIONAL_CREDIT_MAX_EXPECTED_MOVE_RATIO
        and math.isfinite(dte)
        and DIRECTIONAL_CREDIT_MIN_DTE <= dte <= DIRECTIONAL_CREDIT_MAX_DTE
        and math.isfinite(width)
        and width > 0
        and math.isfinite(natural_credit)
        and 0 < natural_credit < width
    )


def _probationary_execution_ready(row: pd.Series | dict[str, Any]) -> bool:
    if _symbol_credit_execution_ready(row) or _walk_forward_credit_execution_ready(row):
        return True
    return bool(
        _probationary_payoff_ready(row)
        and _probationary_confidence_ready(row)
        and _clean(row.get("flow_quality")).lower() == "directional"
        and _clean(row.get("oi_carryover_status")).lower() in {"supportive", "matched_unconfirmed"}
    )


def _payoff_evidence_ready(row: pd.Series | dict[str, Any]) -> bool:
    return (
        _symbol_credit_execution_ready(row)
        or _walk_forward_credit_execution_ready(row)
        or _payoff_model_ready(row)
        or _probationary_payoff_ready(row)
    )


def _payoff_model_ready(row: pd.Series | dict[str, Any]) -> bool:
    def profit_factor_ready(field: str) -> bool:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            return False
        return not math.isnan(value) and value >= V4_EXECUTE_MIN_PROFIT_FACTOR

    minimum_sample = safe_float(row.get("payoff_minimum_sample_required"), 20.0)
    family = "Credit" if _is_credit(row) else ("Debit" if _is_debit(row) else "Unknown")
    direction = _clean(row.get("direction"))
    trend = _clean(row.get("regime_trend") or row.get("regime") or row.get("trend")).lower()
    route_key = _clean(row.get("payoff_route_key") or row.get("payoff_group_key"))
    route_aligned = f"{family}|{direction}|{trend}".lower() in route_key.lower()
    return bool(
        route_aligned
        and family in {"Credit", "Debit"}
        and _clean(row.get("payoff_calibration_status")).upper() == "PASS"
        and safe_float(row.get("payoff_sample_size"), 0.0) >= minimum_sample
        and profit_factor_ready("payoff_stress_10_profit_factor")
        and safe_float(row.get("payoff_walk_forward_oos_sample"), 0.0) >= 5
        and profit_factor_ready("payoff_walk_forward_oos_profit_factor")
        and safe_float(row.get("payoff_post_activation_oos_sample"), 0.0) >= 2
        and safe_float(row.get("payoff_post_activation_oos_average_pnl"), 0.0) > 0
        and profit_factor_ready("payoff_post_activation_oos_profit_factor")
    )


def _medium_debit_sleeve_eligible(row) -> bool:
    """Versioned one-contract bull-call pilot backed by frozen walk-forward evidence."""
    strategy = _clean(row.get("strategy"))
    authorized = _clean(row.get("debit_wf_production_authorized")).lower() in {
        "true", "1", "yes"
    }
    live_guard = _clean(row.get("debit_wf_live_guard_pass")).lower() in {
        "true", "1", "yes"
    }
    probability = safe_float(row.get("debit_wf_predicted_win_probability"), 0.0)
    expected_value = safe_float(row.get("debit_wf_expected_value"), -math.inf)
    prior = safe_float(row.get("debit_wf_prior_sample_size"), 0.0)
    reward_risk = safe_float(row.get("reward_risk"), 0.0)
    quote_width = safe_float(row.get("quote_width_pct"), math.inf)
    oi_status = _clean(row.get("oi_carryover_status")).lower()
    training_through = pd.to_datetime(
        row.get("debit_wf_model_training_through"), errors="coerce"
    )
    signal_value = row.get("v4_asof")
    if not _clean(signal_value) or _clean(signal_value).lower() in {"nan", "nat", "none"}:
        signal_value = row.get("asof")
    if not _clean(signal_value) or _clean(signal_value).lower() in {"nan", "nat", "none"}:
        signal_value = row.get("signal_day")
    signal_day = pd.to_datetime(signal_value, errors="coerce")
    maturity_safe = bool(
        pd.notna(training_through)
        and pd.notna(signal_day)
        and training_through < signal_day
    )
    return bool(
        strategy == "Bull Call Debit Spread"
        and authorized
        and live_guard
        and probability >= 0.55
        and expected_value > 0
        and prior >= 100
        and reward_risk >= V4_MEDIUM_DEBIT_MIN_REWARD_RISK
        and quote_width <= V4_MEDIUM_DEBIT_MAX_QUOTE_WIDTH
        and oi_status in {"supportive", "matched_unconfirmed", "mixed"}
        and maturity_safe
    )


def _effective_win_rate(row: pd.Series | dict[str, Any]) -> float:
    if _symbol_credit_execution_ready(row):
        probability = safe_float(row.get("symbol_credit_wilson_lower_bound"))
        if math.isfinite(probability) and 0 <= probability <= 1:
            return probability
    if _directional_credit_execution_ready(row):
        pooled = safe_float(row.get("directional_credit_execution_wilson_lower_bound"))
        family = safe_float(row.get("directional_credit_family_wilson_lower_bound"))
        probabilities = [value for value in (pooled, family) if math.isfinite(value) and 0 <= value <= 1]
        if probabilities:
            return min(probabilities)
    if _walk_forward_credit_execution_ready(row):
        probability = safe_float(row.get("walk_forward_credit_win_probability"))
        if math.isfinite(probability) and 0 <= probability <= 1:
            return probability
    if _medium_debit_sleeve_eligible(row):
        probability = safe_float(row.get("debit_wf_predicted_win_probability"))
        if math.isfinite(probability) and 0 <= probability <= 1:
            return probability
    payoff_win_rate = safe_float(row.get("payoff_stress_10_win_rate"))
    if _payoff_model_ready(row) and math.isfinite(payoff_win_rate) and 0 <= payoff_win_rate <= 1:
        return payoff_win_rate
    if _probationary_payoff_ready(row) and math.isfinite(payoff_win_rate) and 0 <= payoff_win_rate <= 1:
        lower_bound = safe_float(row.get("confidence_probability_lower_bound"))
        return min(payoff_win_rate, lower_bound) if math.isfinite(lower_bound) else payoff_win_rate
    calibrated = safe_float(row.get("confidence_probability"))
    calibrated_lower = safe_float(row.get("confidence_probability_lower_bound"))
    calibration_status = _clean(row.get("confidence_calibration_status")).upper()
    calibration_tier = _clean(row.get("confidence_model_tier")).lower()
    if math.isfinite(calibrated) and 0 <= calibrated <= 1:
        if calibration_status == "PASS" and calibration_tier.endswith("_validated"):
            if math.isfinite(calibrated_lower) and 0 <= calibrated_lower <= 1:
                return calibrated_lower
        return calibrated
    effective = safe_float(row.get("edge_effective_win_rate"))
    sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size")))
    if math.isfinite(sample) and sample < V4_MIN_REPLAY_SAMPLE:
        return math.nan
    if math.isfinite(effective):
        return effective
    raw = safe_float(row.get("edge_win_rate"))
    if not math.isfinite(raw) or raw < 0 or raw > 1:
        return math.nan
    if math.isfinite(sample) and sample > 0:
        wins = max(0.0, min(sample, raw * sample))
        return (wins + 0.5) / (sample + 1.0)
    return raw


def _reported_win_rate(row: pd.Series | dict[str, Any]) -> float:
    payoff_win_rate = safe_float(row.get("payoff_stress_10_win_rate"))
    if _payoff_evidence_ready(row) and math.isfinite(payoff_win_rate) and 0 <= payoff_win_rate <= 1:
        return _effective_win_rate(row)
    if _medium_debit_sleeve_eligible(row):
        return _effective_win_rate(row)
    calibrated = safe_float(row.get("confidence_probability"))
    if math.isfinite(calibrated) and 0 <= calibrated <= 1:
        return calibrated
    return _effective_win_rate(row)


def _reported_win_rate_basis(row: pd.Series | dict[str, Any]) -> str:
    if _symbol_credit_execution_ready(row):
        return "exact-population symbol credit; 95% Wilson lower bound"
    if _directional_credit_execution_ready(row):
        return "minimum of supportive/matched-OI pooled and direction-family 95% Wilson lower bounds"
    if _walk_forward_credit_execution_ready(row):
        return "trade-specific family model; book validated with maturity-safe 10% fill stress"
    if _payoff_model_ready(row):
        return "validated payoff route; 10% fill stress"
    if _probationary_payoff_ready(row):
        return "confidence lower bound shown; EV uses route 10%-stress outcomes"
    if _medium_debit_sleeve_eligible(row):
        return "validated medium-debit route replay; one-contract authority"
    if math.isfinite(safe_float(row.get("confidence_probability"))):
        return "calibrated strategy-family prior"
    return "historical effective win rate"


def _confidence_evidence_text(row: pd.Series | dict[str, Any]) -> str:
    probability = safe_float(row.get("confidence_probability"))
    lower_bound = safe_float(row.get("confidence_probability_lower_bound"))
    sample = safe_float(row.get("confidence_calibration_sample_size"))
    status = _clean(row.get("confidence_calibration_status")).upper() or "UNAVAILABLE"
    source = _clean(row.get("confidence_probability_source")) or "unavailable"
    if not math.isfinite(probability):
        return "unavailable"
    lower_text = f", 90% lower bound {lower_bound:.0%}" if math.isfinite(lower_bound) else ""
    sample_text = f", n={int(sample)}" if math.isfinite(sample) else ""
    return f"{source} {probability:.0%}{lower_text}{sample_text}, {status}"


def _confidence_rating(row: pd.Series | dict[str, Any]) -> str:
    conservative_win = _effective_win_rate(row)
    win_text = f" · {conservative_win:.0%} floor" if math.isfinite(conservative_win) else ""
    if _directional_credit_execution_ready(row) or _walk_forward_credit_execution_ready(row):
        return f"🟡 MEDIUM{win_text}"
    if _symbol_credit_execution_ready(row):
        return f"🟠 PILOT{win_text}"
    if _payoff_model_ready(row) and confidence_high_ready(row):
        return f"🟢 HIGH{win_text}"
    if _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row):
        return f"🟡 MEDIUM{win_text}"
    research_win = _reported_win_rate(row)
    research_text = f" · {research_win:.0%} model estimate" if math.isfinite(research_win) else ""
    return f"⚪ RESEARCH ONLY{research_text}"


def _edge_evidence_text(row: pd.Series | dict[str, Any]) -> str:
    verdict = _clean(row.get("edge_verdict")).lower() or "unavailable"
    sample = safe_float(row.get("edge_sample_size"))
    win = safe_float(row.get("edge_win_rate"))
    profit_factor = safe_float(row.get("edge_profit_factor"))
    average = safe_float(row.get("edge_avg_pnl"))
    match = _clean(row.get("edge_match_level")) or "unavailable"
    if not math.isfinite(sample) or sample <= 0:
        if _payoff_evidence_ready(row):
            return f"{verdict} exact-match edge; route evidence shown separately"
        return f"{verdict} (no per-ticket replay sample; family-prior only)"
    pieces = [f"{verdict}", f"n={int(sample)}", f"match={match}"]
    if math.isfinite(win):
        pieces.append(f"win {win:.0%}")
    if math.isfinite(profit_factor):
        pieces.append(f"PF {profit_factor:.2f}")
    if math.isfinite(average):
        pieces.append(f"avg ${average:.0f}")
    return ", ".join(pieces)


def _profit_factor_text(value: object) -> str:
    try:
        profit_factor = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isinf(profit_factor) and profit_factor > 0:
        return "inf (no historical stressed losses)"
    if math.isfinite(profit_factor):
        return f"{profit_factor:.2f}"
    return "n/a"


def _payoff_evidence_text(row: pd.Series | dict[str, Any]) -> str:
    if _symbol_credit_execution_ready(row):
        return (
            f"SYMBOL-TREND CREDIT VALIDATED; group={_clean(row.get('symbol_credit_group_key'))}; "
            f"n={int(safe_float(row.get('symbol_credit_sample_size'), 0.0))}; "
            f"conservative win={safe_float(row.get('symbol_credit_wilson_lower_bound'), 0.0):.0%}; "
            f"Bayesian mean={safe_float(row.get('symbol_credit_bayesian_win_probability'), 0.0):.0%}; "
            f"10% fill-stress PF={_profit_factor_text(row.get('symbol_credit_stress_profit_factor_10pct'))}; "
            f"holdout n={int(safe_float(row.get('symbol_credit_oos_sample_size'), 0.0))}; "
            f"episodes={int(safe_float(row.get('symbol_credit_independent_episode_count'), 0.0))}; "
            f"post-live n={int(safe_float(row.get('symbol_credit_postactivation_sample_size'), 0.0))}"
        )
    if _directional_credit_execution_ready(row):
        return (
            "DIRECTIONAL CREDIT MEDIUM; "
            f"execution-policy n={int(safe_float(row.get('directional_credit_execution_sample_size'), 0.0))}; "
            f"95% Wilson floor={safe_float(row.get('directional_credit_execution_wilson_lower_bound'), 0.0):.0%}; "
            f"10% fill-stress PF={_profit_factor_text(row.get('directional_credit_execution_stress_profit_factor_10pct'))}; "
            f"holdout n={int(safe_float(row.get('directional_credit_execution_holdout_sample_size'), 0.0))}; "
            f"holdout win={safe_float(row.get('directional_credit_execution_holdout_win_rate'), 0.0):.0%}; "
            f"holdout PF={_profit_factor_text(row.get('directional_credit_execution_holdout_stress_profit_factor_10pct'))}; "
            f"family={_clean(row.get('directional_credit_family_validation_status')).upper()}, "
            f"family n={int(safe_float(row.get('directional_credit_family_sample_size'), 0.0))}, "
            f"family PF={_profit_factor_text(row.get('directional_credit_family_stress_profit_factor_10pct'))}, "
            f"family holdout n={int(safe_float(row.get('directional_credit_family_holdout_sample_size'), 0.0))}, "
            f"family holdout PF={_profit_factor_text(row.get('directional_credit_family_holdout_stress_profit_factor_10pct'))}; "
            "one contract; High confidence unavailable"
        )
    if _walk_forward_credit_execution_ready(row):
        return (
            "WALK-FORWARD CREDIT MEDIUM; "
            f"trade win probability={safe_float(row.get('walk_forward_credit_win_probability'), 0.0):.0%}; "
            f"predicted 10%-stress ROR={safe_float(row.get('walk_forward_credit_prediction'), 0.0):.1%}; "
            f"OOS n={int(safe_float(row.get('walk_forward_credit_oos_sample_size'), 0.0))}; "
            f"wins={int(safe_float(row.get('walk_forward_credit_oos_wins'), 0.0))}; "
            f"Bayesian win={safe_float(row.get('walk_forward_credit_bayesian_win_probability'), 0.0):.0%}; "
            f"95% Wilson floor={safe_float(row.get('walk_forward_credit_wilson_lower_bound'), 0.0):.0%}; "
            f"10% fill-stress PF={_profit_factor_text(row.get('walk_forward_credit_stress_profit_factor_10pct'))}; "
            f"positive validation months={int(safe_float(row.get('walk_forward_credit_positive_months'), 0.0))}; "
            "one contract; High confidence unavailable"
        )
    if _medium_debit_sleeve_eligible(row):
        policy = _clean(row.get("debit_wf_policy_version")) or "frozen walk-forward debit policy"
        prior_sample = safe_float(row.get("debit_wf_prior_sample_size"))
        predicted_win = safe_float(row.get("debit_wf_predicted_win_probability"))
        expected_value = safe_float(row.get("debit_wf_expected_value"))
        training_through = _clean(row.get("debit_wf_model_training_through")) or "unavailable"
        live_guard = _clean(row.get("debit_wf_live_guard_pass")).lower() in {"true", "1", "yes"}
        return (
            f"VALIDATED MEDIUM-DEBIT ROUTE; policy={policy}; "
            + (f"prior n={int(prior_sample)}; " if math.isfinite(prior_sample) else "")
            + (f"predicted win={predicted_win:.0%}; " if math.isfinite(predicted_win) else "")
            + (f"post-pricing EV={_money(expected_value)}; " if math.isfinite(expected_value) else "")
            + f"training through={training_through}; live guard={'PASS' if live_guard else 'FAIL'}; "
            "one contract; High confidence unavailable"
        )
    status = _clean(row.get("payoff_calibration_status")).upper() or "FAIL"
    sample = safe_float(row.get("payoff_sample_size"))
    win_rate = safe_float(row.get("payoff_stress_10_win_rate"))
    profit_factor = safe_float(row.get("payoff_stress_10_profit_factor"))
    average_pnl = safe_float(row.get("payoff_stress_10_average_pnl"))
    oos_sample = safe_float(row.get("payoff_walk_forward_oos_sample"))
    oos_pf = safe_float(row.get("payoff_walk_forward_oos_profit_factor"))
    pieces = [status]
    if math.isfinite(sample):
        pieces.append(f"n={int(sample)}")
    if math.isfinite(win_rate):
        pieces.append(f"10% fill-stress win={win_rate:.0%}")
    if math.isfinite(profit_factor):
        pieces.append(f"10% fill-stress PF={profit_factor:.2f}")
    if math.isfinite(average_pnl):
        pieces.append(f"avg={_money(average_pnl)}")
    if math.isfinite(oos_sample):
        pieces.append(f"OOS n={int(oos_sample)}")
    if math.isfinite(oos_pf):
        pieces.append(f"OOS PF={oos_pf:.2f}")
    return ", ".join(pieces)


def build_execution_evidence_integrity(scored: pd.DataFrame | None) -> dict[str, Any]:
    if scored is None or scored.empty:
        return {
            "status": "FAIL",
            "reason": "candidate book is empty",
            "minimum_specific_sample": V4_MIN_REPLAY_SAMPLE,
            "maximum_specific_sample": 0,
            "validated_family_rows": 0,
            "evidence_reachable_rows": 0,
        }
    samples = pd.to_numeric(scored.get("edge_sample_size", pd.Series(index=scored.index, dtype=float)), errors="coerce")
    family_mask = scored.apply(_validated_family_evidence, axis=1)
    probationary_mask = scored.apply(_probationary_execution_ready, axis=1)
    symbol_credit_mask = scored.apply(_symbol_credit_execution_ready, axis=1)
    walk_forward_credit_mask = scored.apply(_walk_forward_credit_execution_ready, axis=1)
    medium_debit_mask = scored.apply(_medium_debit_sleeve_eligible, axis=1)
    payoff_mask = scored.get("payoff_calibration_status", pd.Series("FAIL", index=scored.index)).astype(str).str.upper().eq("PASS")
    specific_mask = samples.ge(V4_MIN_REPLAY_SAMPLE)
    reachable = (
        specific_mask
        | family_mask
        | probationary_mask
        | symbol_credit_mask
        | walk_forward_credit_mask
        | medium_debit_mask
    )
    maximum = int(samples.max()) if samples.notna().any() else 0
    reachable_count = int(reachable.sum())
    return {
        "status": "PASS" if reachable_count else "FAIL",
        "reason": (
            "specific or validated family evidence can authorize execution"
            if reachable_count
            else "no validated realized-payoff lane matches the current strategy, direction, and regime"
            if not payoff_mask.any()
            else "configured specific-history minimum exceeds all available samples and no confidence-calibrated family fallback exists"
        ),
        "minimum_specific_sample": V4_MIN_REPLAY_SAMPLE,
        "maximum_specific_sample": maximum,
        "specific_evidence_rows": int(specific_mask.sum()),
        "validated_payoff_lane_rows": int(payoff_mask.sum()),
        "validated_family_rows": int(family_mask.sum()),
        "probationary_evidence_rows": int(probationary_mask.sum()),
        "symbol_credit_evidence_rows": int(symbol_credit_mask.sum()),
        "walk_forward_credit_evidence_rows": int(walk_forward_credit_mask.sum()),
        "medium_debit_evidence_rows": int(medium_debit_mask.sum()),
        "evidence_reachable_rows": reachable_count,
    }


def build_execution_reachability_audit(scored: pd.DataFrame | None) -> dict[str, Any]:
    """Explain whether zero Execute is structural, quality-driven, or price-driven."""
    if scored is None or scored.empty:
        return {
            "status": "FAIL",
            "reason": "candidate book is empty",
            "candidate_rows": 0,
            "evidence_authorized_rows": 0,
            "approved_target_rows": 0,
            "entry_ready_before_same_ticker_selection": 0,
            "final_execute_rows": 0,
        }

    counts = {
        "candidate_rows": int(len(scored)),
        "evidence_authorized_rows": 0,
        "hard_risk_clear_rows": 0,
        "safety_clear_rows": 0,
        "positive_expectancy_rows": 0,
        "defined_risk_rows": 0,
        "non_price_quality_clear_rows": 0,
        "approved_target_rows": 0,
        "current_entry_target_met_rows": 0,
        "immediate_quote_width_clear_rows": 0,
        "entry_ready_before_same_ticker_selection": 0,
        "price_wait_rows": 0,
        "quote_wait_rows": 0,
        "session_reprice_rows": 0,
        "catalyst_verification_wait_rows": 0,
        "final_execute_rows": int(
            scored.get("trade_status", pd.Series("", index=scored.index)).astype(str).eq("Execute").sum()
        ),
        "market_open_rows": int(
            scored.get("market_session_open_at_validation", pd.Series(False, index=scored.index))
            .map(_truthy_flag)
            .sum()
        ),
    }
    route_counts = {
        "directional_credit_medium": 0,
        "strict_walk_forward_credit": 0,
        "symbol_credit_pilot": 0,
        "validated_medium_debit": 0,
        "other_validated_payoff": 0,
    }
    blocker_counts: dict[str, int] = {}

    for _, row in scored.iterrows():
        directional = _directional_credit_execution_ready(row)
        symbol_credit = _symbol_credit_execution_ready(row)
        walk_forward = _walk_forward_credit_execution_ready(row)
        medium_debit = _medium_debit_sleeve_eligible(row)
        other_payoff = _payoff_model_ready(row) or _validated_family_evidence(row)
        authorized = directional or symbol_credit or walk_forward or medium_debit or other_payoff
        if not authorized:
            continue
        counts["evidence_authorized_rows"] += 1
        if directional:
            route_counts["directional_credit_medium"] += 1
        elif symbol_credit:
            route_counts["symbol_credit_pilot"] += 1
        elif walk_forward:
            route_counts["strict_walk_forward_credit"] += 1
        elif medium_debit:
            route_counts["validated_medium_debit"] += 1
        else:
            route_counts["other_validated_payoff"] += 1

        hard_clear = not bool(_hard_blocker_reason(row))
        safety_clear = not bool(_safety_research_reason(row))
        ev_clear = _v4_nonnegative_ev(row)
        risk_clear = bool(
            math.isfinite(_max_loss_value(row))
            and _max_loss_value(row) > 0
            and math.isfinite(_target_profit_value(row))
        )
        execution_quote_blocker = _execution_quote_blocker(row)
        execution_catalyst_blocker = _execution_catalyst_verification_blocker(row)
        identification_blocker = _execute_quality_blocker(
            row,
            include_entry_price=False,
            include_execution_quote=False,
            include_catalyst_verification=False,
        )
        quality_clear = not bool(identification_blocker)
        target_met = _v4_target_met(row)
        quote_width = safe_float(row.get("quote_width_pct"), 9.0)
        combination_width = _combination_quote_width_ratio(row)
        if math.isfinite(combination_width):
            quote_width = max(quote_width, combination_width)
        quote_clear = quote_width <= 0.20
        counts["hard_risk_clear_rows"] += int(hard_clear)
        counts["safety_clear_rows"] += int(safety_clear)
        counts["positive_expectancy_rows"] += int(ev_clear)
        counts["defined_risk_rows"] += int(risk_clear)
        counts["non_price_quality_clear_rows"] += int(quality_clear)
        counts["current_entry_target_met_rows"] += int(target_met)
        counts["immediate_quote_width_clear_rows"] += int(quote_clear)
        approved = hard_clear and safety_clear and ev_clear and risk_clear and quality_clear
        counts["approved_target_rows"] += int(approved)
        counts["price_wait_rows"] += int(approved and not target_met)
        counts["quote_wait_rows"] += int(
            approved and target_met and (not quote_clear or bool(execution_quote_blocker))
        )
        counts["session_reprice_rows"] += int(approved and bool(execution_quote_blocker))
        counts["catalyst_verification_wait_rows"] += int(
            approved and bool(execution_catalyst_blocker)
        )
        counts["entry_ready_before_same_ticker_selection"] += int(
            approved
            and target_met
            and quote_clear
            and not execution_quote_blocker
            and not execution_catalyst_blocker
        )
        if not quality_clear:
            blocker_counts[identification_blocker] = blocker_counts.get(identification_blocker, 0) + 1

    approved = counts["approved_target_rows"]
    final_execute = counts["final_execute_rows"]
    if not counts["evidence_authorized_rows"]:
        status = "FAIL"
        reason = "no evidence-authorized execution route matches the current universe"
    elif not approved:
        status = "BLOCKED"
        reason = "evidence routes exist, but every row fails a non-price execution-quality gate"
    elif final_execute:
        status = "PASS"
        reason = "execution is reachable and at least one current structure clears every gate"
    else:
        status = "PASS_PRICE_WAIT"
        reason = (
            "approved structures exist; current entry price, executable quote width, a fresh "
            "regular-session reprice, or same-session catalyst verification is the remaining wait"
        )
    return {
        "status": status,
        "reason": reason,
        **counts,
        "route_authority_counts": route_counts,
        "non_price_blocker_counts": dict(
            sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "market_session_affects_identification": False,
        "identification_policy": (
            "market-open state never removes an approved setup; current/last regular-session option liquidity, "
            "entry price, and a fresh pre-order quote control Enter/Execute"
        ),
        "arbitrary_signal_cap": None,
    }


def _walk_forward_credit_expectancy(
    row: pd.Series | dict[str, Any],
    risk: float,
) -> tuple[float, float, float, float]:
    if _walk_forward_credit_execution_ready(row):
        if _directional_credit_execution_ready(row):
            predicted_return = safe_float(
                row.get("directional_credit_family_stress_average_return_on_risk_10pct")
            )
            profit_factor = safe_float(
                row.get("directional_credit_family_stress_profit_factor_10pct")
            )
            win_rate = _effective_win_rate(row)
        else:
            predicted_return = safe_float(row.get("walk_forward_credit_prediction"))
            profit_factor = safe_float(row.get("walk_forward_credit_stress_profit_factor_10pct"))
            win_rate = safe_float(row.get("walk_forward_credit_win_probability"), _effective_win_rate(row))
        if (
            math.isfinite(predicted_return)
            and predicted_return > 0
            and math.isfinite(profit_factor)
            and profit_factor > 1
            and math.isfinite(win_rate)
            and 0 < win_rate < 1
            and math.isfinite(risk)
            and risk > 0
        ):
            expected_value = predicted_return * risk
            gross_losses = expected_value / (profit_factor - 1.0)
            gross_wins = profit_factor * gross_losses
            return (
                expected_value,
                profit_factor,
                gross_wins / win_rate,
                gross_losses / (1.0 - win_rate),
            )
    return math.nan, math.nan, math.nan, math.nan


def _medium_debit_economics_at_price(
    row: pd.Series | dict[str, Any],
    debit: float,
) -> tuple[float, float, float, float, float, float]:
    """Return target, nominal risk, EV, PF, stressed win, and stressed loss."""
    probability = safe_float(row.get("debit_wf_predicted_win_probability"))
    width = safe_float(row.get("spread_width"))
    if (
        not _medium_debit_sleeve_eligible(row)
        or not math.isfinite(probability)
        or not 0 < probability < 1
        or not math.isfinite(debit)
        or debit <= 0
        or not math.isfinite(width)
        or width <= debit
    ):
        return (math.nan,) * 6
    target_value = min(
        width,
        debit * (1.0 + V4_MEDIUM_DEBIT_PROFIT_TAKE_PCT),
    )
    target_profit = (target_value - debit) * 100.0
    max_loss = debit * 100.0
    stressed_cost = debit * (1.0 + V4_MEDIUM_DEBIT_ENTRY_STRESS_PCT) * 100.0
    win_payoff = max(0.0, target_value * 100.0 - stressed_cost)
    loss_payoff = stressed_cost
    gross_wins = probability * win_payoff
    gross_losses = (1.0 - probability) * loss_payoff
    expected_value = gross_wins - gross_losses
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else math.inf
    return target_profit, max_loss, expected_value, profit_factor, win_payoff, loss_payoff


def _medium_debit_ticket_economics(
    row: pd.Series | dict[str, Any],
) -> tuple[float, float, float, float, float, float]:
    safe_price = _expectancy_safe_entry_price(row)
    current_price = _v4_current_entry_price(row)
    entry = safe_price
    if (
        math.isfinite(current_price)
        and current_price > 0
        and math.isfinite(safe_price)
        and current_price <= safe_price
    ):
        entry = current_price
    return _medium_debit_economics_at_price(row, entry)


def _post_pricing_expectancy(
    row: pd.Series | dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return coherent EV, PF, win payoff, and loss payoff for the final structure."""
    if _medium_debit_sleeve_eligible(row):
        _, _, expected_value, profit_factor, win_payoff, loss_payoff = (
            _medium_debit_economics_at_price(row, _v4_current_entry_price(row))
        )
        if math.isfinite(expected_value):
            return expected_value, profit_factor, win_payoff, loss_payoff
    walk_forward = _walk_forward_credit_expectancy(row, _max_loss_value(row))
    if math.isfinite(walk_forward[0]):
        return walk_forward
    if _payoff_evidence_ready(row):
        win_rate = safe_float(row.get("payoff_stress_10_win_rate"))
        win_fraction = safe_float(row.get("payoff_stress_10_average_win_risk_fraction"))
        loss_fraction = safe_float(row.get("payoff_stress_10_average_loss_risk_fraction"))
        risk = _max_loss_value(row)
        if (
            math.isfinite(win_rate)
            and 0 < win_rate < 1
            and math.isfinite(win_fraction)
            and win_fraction > 0
            and math.isfinite(loss_fraction)
            and loss_fraction > 0
            and math.isfinite(risk)
            and risk > 0
        ):
            win_payoff = win_fraction * risk
            loss_payoff = loss_fraction * risk
            gross_wins = win_rate * win_payoff
            gross_losses = (1.0 - win_rate) * loss_payoff
            return (
                gross_wins - gross_losses,
                gross_wins / gross_losses if gross_losses > 0 else math.inf,
                win_payoff,
                loss_payoff,
            )
    win_rate = _effective_win_rate(row)
    win_payoff = _target_profit_value(row)
    loss_payoff = _max_loss_value(row)
    if (
        not math.isfinite(win_rate)
        or win_rate <= 0
        or win_rate > 1
        or not math.isfinite(win_payoff)
        or win_payoff <= 0
        or not math.isfinite(loss_payoff)
        or loss_payoff <= 0
    ):
        return math.nan, math.nan, win_payoff, loss_payoff
    gross_wins = win_rate * win_payoff
    gross_losses = (1.0 - win_rate) * loss_payoff
    expected_value = gross_wins - gross_losses
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else math.inf
    return expected_value, profit_factor, win_payoff, loss_payoff


def _post_pricing_expectancy_blocker(row: pd.Series | dict[str, Any]) -> str:
    expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
    if not math.isfinite(expected_value) or math.isnan(profit_factor):
        return "post-pricing expectancy unavailable for the final quoted structure"
    if expected_value <= 0:
        return f"post-pricing expected value is ${expected_value:.2f}; must be above $0"
    if profit_factor < V4_EXECUTE_MIN_PROFIT_FACTOR:
        return (
            f"post-pricing profit factor is {profit_factor:.2f}; "
            f"must be at least {V4_EXECUTE_MIN_PROFIT_FACTOR:.2f}"
        )
    return ""


def _expectancy_safe_entry_price(row: pd.Series | dict[str, Any]) -> float:
    # A non-validated payoff lane may be useful for research, but it must not
    # manufacture an "expectancy-safe" executable price from proxy math.
    if not _payoff_evidence_ready(row) and not _medium_debit_sleeve_eligible(row):
        return math.nan
    win_rate = _effective_win_rate(row)
    width = safe_float(row.get("spread_width"))
    max_profit = safe_float(row.get("max_profit"))
    win_payoff = _target_profit_value(row)
    configured_target = safe_float(
        row.get("required_entry"),
        safe_float(row.get("target_entry"), safe_float(row.get("entry_target"))),
    )
    if _medium_debit_sleeve_eligible(row):
        # The frozen model already stress-prices the exact natural debit by
        # 10%. Keep the configured no-chase limit instead of replacing it with
        # an unrelated generic payoff-route ceiling.
        if math.isfinite(configured_target) and configured_target > 0:
            return math.floor(configured_target * 100.0 + 1e-9) / 100.0
        return math.nan
    if (
        not math.isfinite(win_rate)
        or win_rate <= 0
        or win_rate >= 1
        or not math.isfinite(width)
        or width <= 0
        or not math.isfinite(max_profit)
        or max_profit <= 0
        or not math.isfinite(win_payoff)
        or win_payoff <= 0
    ):
        return math.nan
    if _directional_credit_execution_ready(row):
        # This lane was replayed and holdout-tested on its own 15%-45% credit
        # band.  Its configured target is already inside that contract.  The
        # former generic PF algebra assumed a different payoff route and
        # overwrote every target with an untested ~35%-of-width floor.
        floor = width * DIRECTIONAL_CREDIT_MIN_PCT_WIDTH
        if math.isfinite(configured_target) and configured_target > 0:
            floor = max(floor, configured_target)
        if floor > width * DIRECTIONAL_CREDIT_MAX_PCT_WIDTH:
            return math.nan
        return math.ceil(floor * 100.0 - 1e-9) / 100.0
    if _payoff_evidence_ready(row) and not _directional_credit_execution_ready(row):
        route_level = _clean(row.get("payoff_route_level")).lower()
        route_key = _clean(row.get("payoff_route_key") or row.get("payoff_group_key")).lower()
        cost_calibrated_route = route_level == "flow_cost" and "cost=" in route_key
        if _is_credit(row):
            empirical_fraction = safe_float(row.get("payoff_entry_pct_width_p25"))
            empirical_price = width * empirical_fraction if math.isfinite(empirical_fraction) else math.nan
            if _payoff_model_ready(row) and cost_calibrated_route:
                floor = empirical_price if math.isfinite(empirical_price) else configured_target
            else:
                floor = configured_target if math.isfinite(configured_target) and configured_target > 0 else 0.0
                if math.isfinite(empirical_price):
                    floor = max(floor, empirical_price)
            return math.ceil(floor * 100.0 - 1e-9) / 100.0 if floor > 0 else math.nan
        empirical_fraction = safe_float(row.get("payoff_entry_pct_width_p75"))
        empirical_price = width * empirical_fraction if math.isfinite(empirical_fraction) else math.nan
        if _payoff_model_ready(row) and cost_calibrated_route:
            ceiling = empirical_price if math.isfinite(empirical_price) else configured_target
        else:
            ceiling = configured_target if math.isfinite(configured_target) and configured_target > 0 else width
            if math.isfinite(empirical_price):
                ceiling = min(ceiling, empirical_price)
        return math.floor(ceiling * 100.0 + 1e-9) / 100.0 if ceiling > 0 else math.nan
    payoff_fraction = max(0.01, min(1.0, win_payoff / max_profit))
    loss_rate = 1.0 - win_rate
    if _is_credit(row):
        denominator = win_rate * payoff_fraction + V4_EXECUTE_MIN_PROFIT_FACTOR * loss_rate
        if denominator <= 0:
            return math.nan
        expectancy_floor = V4_EXECUTE_MIN_PROFIT_FACTOR * loss_rate * width / denominator
        policy_floor = width * 0.18
        if math.isfinite(configured_target):
            policy_floor = max(policy_floor, configured_target)
        return math.ceil(max(policy_floor, expectancy_floor) * 100.0 - 1e-9) / 100.0
    if _is_debit(row):
        denominator = win_rate * payoff_fraction + V4_EXECUTE_MIN_PROFIT_FACTOR * loss_rate
        if denominator <= 0:
            return math.nan
        expectancy_ceiling = win_rate * payoff_fraction * width / denominator
        policy_ceiling = configured_target if math.isfinite(configured_target) and configured_target > 0 else width
        return math.floor(min(policy_ceiling, expectancy_ceiling) * 100.0 + 1e-9) / 100.0
    return math.nan


def _expectancy_safe_entry_target(row: pd.Series | dict[str, Any]) -> str:
    if not _payoff_evidence_ready(row) and not _medium_debit_sleeve_eligible(row):
        review_target = _entry_target(row)
        return f"REVIEW ONLY - {review_target}; no execution authority"
    safe_price = _expectancy_safe_entry_price(row)
    if not math.isfinite(safe_price):
        return _entry_target(row)
    return f">= ${safe_price:.2f} credit" if _is_credit(row) else f"<= ${safe_price:.2f} debit"


def _ticket_entry_target(row: pd.Series | dict[str, Any], disposition: str) -> str:
    safe_target = _expectancy_safe_entry_target(row)
    debit_evidence_ready = _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row)
    if not _is_debit(row) or not debit_evidence_ready:
        return safe_target
    current = _v4_current_entry_price(row)
    safe_price = _expectancy_safe_entry_price(row)
    if not math.isfinite(current) or current <= 0:
        return safe_target
    if not math.isfinite(safe_price) or current > safe_price:
        return safe_target
    current_target = f"<= ${current:.2f} debit; do not chase above ${safe_price:.2f}"
    if disposition == "Scout":
        return f"REVIEW ONLY - {current_target}; no execution authority"
    return current_target


def _entry_limit_expectancy(
    row: pd.Series | dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return conservative economics at the stated minimum/maximum entry limit."""
    safe_price = _expectancy_safe_entry_price(row)
    current_price = _v4_current_entry_price(row)
    # Debit tickets explicitly quote the current favorable debit as the order
    # target and separately show the no-chase ceiling. Credit tickets quote a
    # minimum acceptable credit, so their risk must remain based on that floor.
    if (
        _is_debit(row)
        and math.isfinite(current_price)
        and current_price > 0
        and math.isfinite(safe_price)
        and current_price <= safe_price
    ):
        safe_price = current_price
    if _medium_debit_sleeve_eligible(row):
        target_profit, max_loss, expected_value, profit_factor, _, _ = (
            _medium_debit_economics_at_price(row, safe_price)
        )
        if math.isfinite(expected_value):
            return target_profit, max_loss, expected_value, profit_factor
    width = safe_float(row.get("spread_width"))
    current_max_profit = safe_float(row.get("max_profit"))
    current_target_profit = _target_profit_value(row)
    win_rate = _effective_win_rate(row)
    if (
        not math.isfinite(safe_price)
        or not math.isfinite(width)
        or safe_price <= 0
        or safe_price >= width
        or not math.isfinite(current_max_profit)
        or current_max_profit <= 0
        or not math.isfinite(current_target_profit)
        or current_target_profit <= 0
        or not math.isfinite(win_rate)
    ):
        expected_value, profit_factor, win_payoff, loss_payoff = _post_pricing_expectancy(row)
        return win_payoff, loss_payoff, expected_value, profit_factor
    target_fraction = max(0.01, min(1.0, current_target_profit / current_max_profit))
    if _is_credit(row):
        max_profit = safe_price * 100.0
        max_loss = (width - safe_price) * 100.0
    else:
        max_profit = (width - safe_price) * 100.0
        max_loss = safe_price * 100.0
    target_profit = max_profit * target_fraction
    walk_forward = _walk_forward_credit_expectancy(row, max_loss)
    if math.isfinite(walk_forward[0]):
        return target_profit, max_loss, walk_forward[0], walk_forward[1]
    if _payoff_evidence_ready(row):
        payoff_win_rate = safe_float(row.get("payoff_stress_10_win_rate"))
        win_fraction = safe_float(row.get("payoff_stress_10_average_win_risk_fraction"))
        loss_fraction = safe_float(row.get("payoff_stress_10_average_loss_risk_fraction"))
        if (
            math.isfinite(payoff_win_rate)
            and 0 < payoff_win_rate < 1
            and math.isfinite(win_fraction)
            and win_fraction > 0
            and math.isfinite(loss_fraction)
            and loss_fraction > 0
        ):
            gross_wins = payoff_win_rate * win_fraction * max_loss
            gross_losses = (1.0 - payoff_win_rate) * loss_fraction * max_loss
            return (
                target_profit,
                max_loss,
                gross_wins - gross_losses,
                gross_wins / gross_losses if gross_losses > 0 else math.inf,
            )
    gross_wins = win_rate * target_profit
    gross_losses = (1.0 - win_rate) * max_loss
    expected_value = gross_wins - gross_losses
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else math.inf
    return target_profit, max_loss, expected_value, profit_factor


def _expectancy_safe_price_blocker(row: pd.Series | dict[str, Any]) -> str:
    safe_price = _expectancy_safe_entry_price(row)
    if not math.isfinite(safe_price):
        return "expectancy-safe entry limit is unavailable"
    current = _v4_current_entry_price(row)
    if not math.isfinite(current):
        return "current executable spread price is unavailable"
    if _is_credit(row) and current + 1e-9 < safe_price:
        return f"natural credit ${current:.2f} is below expectancy-safe minimum ${safe_price:.2f}; work the limit, do not accept less"
    if _is_debit(row) and current - 1e-9 > safe_price:
        return f"natural debit ${current:.2f} is above expectancy-safe maximum ${safe_price:.2f}; work the limit, do not chase"
    return ""


def _blocker_text(row: pd.Series | dict[str, Any]) -> str:
    if _clean(row.get("trade_status")) == "Execute" and not _is_hard_blocked(row) and not _execute_quality_blocker(row):
        catalyst_status = _clean(row.get("catalyst_status")).lower()
        if not is_etf_row(pd.Series(row)) and catalyst_status in {"", "unknown"}:
            event_date = earnings_event_date(row)
            earnings_context = (
                f"; next structured earnings date is {event_date}"
                if event_date is not None
                else "; earnings date is unresolved"
            )
            return (
                "Pre-order evidence condition: ticker-specific news/catalyst status is unconfirmed"
                f"{earnings_context}; complete a same-session company-news and earnings check, then enter only "
                "at/inside target with attached OCO after a fresh quote."
            )
        return "No hard blocker; enter only at/inside target with attached OCO after same-session quote refresh."
    safe_price_blocker = _expectancy_safe_price_blocker(row)
    if "expectancy-safe" in safe_price_blocker and ("below" in safe_price_blocker or "above" in safe_price_blocker):
        return safe_price_blocker
    if _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row):
        # Once a current evidence route is authorized, legacy discovery fields
        # such as ``what_must_improve`` and ``primary_blocker`` are historical
        # context, not the binding reason for the final disposition.
        direct_reason = _clean(row.get("v4_direct_disposition_reason"))
        if direct_reason:
            return direct_reason
        current_reason = _clean(row.get("trade_status_reason"))
        if current_reason:
            return current_reason
    for key in ["what_must_improve", "primary_blocker", "trade_status_reason", "hard_rejects", "penalties"]:
        text = _clean(row.get(key))
        if text:
            return text
    return "fresh Schwab quote, news, OI/flow, and portfolio-risk confirmation"


def _disposition(row: pd.Series | dict[str, Any], *, targetable: bool | None = None) -> str:
    if _is_hard_blocked(row):
        return "Avoid"
    if _safety_research_reason(row):
        return "Research"
    strategy = f"{row.get('strategy', '')} {row.get('Trade', '')}".lower()
    if "cash-secured" in strategy or "wheel" in strategy:
        return "Wheel/Cash"
    status = _clean(row.get("trade_status"))
    tier = _clean(row.get("trade_tier")).lower()
    if status == "Execute" and "pilot" in tier and _probationary_execution_ready(row):
        return "Execute"
    if (
        status == "Execute"
        and _clean(row.get("v4_execution_authority")) == "validated_medium_debit_one_lot"
        and _medium_debit_sleeve_eligible(row)
    ):
        return "Execute"
    if status == "Watch" and tier == "scout":
        return "Scout"
    if not _payoff_evidence_ready(row) and not _medium_debit_sleeve_eligible(row):
        return "Research"
    if not status:
        status = "Execute" if "Execute" in _clean(row.get("Status")) else "Watch" if "Scout" in _clean(row.get("Status")) else ""
    if status == "Execute":
        return "Execute"
    if status == "Watch" and ("work-limit" in tier or "near-trigger" in tier or "price" in tier):
        return "Swing Target / Work Limit"
    if status == "Watch":
        return "Scout"
    if targetable:
        return "Swing Target / Work Limit"
    if status == "Avoid":
        return "Avoid"
    return "Research"


def _targetable(row: pd.Series | dict[str, Any]) -> bool:
    if _is_hard_blocked(row):
        return False
    if _safety_research_reason(row):
        return False
    if not _v4_nonnegative_ev(row):
        return False
    entry_target = _entry_target(row).lower()
    if "$" not in entry_target or not any(kind in entry_target for kind in ("credit", "debit")):
        return False
    status = _clean(row.get("trade_status"))
    if status in {"Execute", "Watch"}:
        return True
    if _clean(row.get("live_status")) not in {"PASS", "pass", ""}:
        return False
    if not math.isfinite(_target_profit_value(row)) or not math.isfinite(_max_loss_value(row)):
        return False
    validated_one_lot_lane = (
        _symbol_credit_execution_ready(row)
        or _walk_forward_credit_execution_ready(row)
        or _medium_debit_sleeve_eligible(row)
    )
    if validated_one_lot_lane:
        return safe_float(row.get("quote_width_pct"), 9.0) <= 0.50
    score = safe_float(row.get("score"), 0.0)
    confirmation = safe_float(row.get("confirmation_score"), 0.0)
    quote_width = safe_float(row.get("quote_width_pct"), 0.0)
    replay = _clean(row.get("replay_ev_verdict")).lower()
    edge = _clean(row.get("edge_verdict")).lower()
    negative_edge = replay.startswith("negative") or edge == "negative"
    if negative_edge and edge not in {"thin_sample", "acceptable", "positive"}:
        return False
    return score >= 4.5 and confirmation >= 5.0 and quote_width <= 0.50


def _trend_evidence(row: pd.Series | dict[str, Any]) -> str:
    parts = []
    direction = _clean(row.get("direction"))
    if direction:
        parts.append(direction)
    for label, key in [
        ("flow", "flow_quality"),
        ("edge", "edge_verdict"),
        ("score", "score"),
        ("confirm", "confirmation_score"),
    ]:
        value = row.get(key)
        if label in {"score", "confirm"}:
            number = safe_float(value)
            if math.isfinite(number):
                parts.append(f"{label}={number:.1f}")
        else:
            text = _clean(value)
            if text:
                parts.append(f"{label}={text}")
    return "; ".join(parts)


def _flow_oi_evidence(row: pd.Series | dict[str, Any], top_flow: pd.DataFrame) -> str:
    ticker = _clean(row.get("ticker") or row.get("Ticker")).upper()
    parts = []
    if not top_flow.empty and "ticker" in top_flow.columns:
        hit = top_flow[top_flow["ticker"].astype(str).str.upper().eq(ticker)]
        if not hit.empty:
            first = hit.iloc[0]
            parts.append(f"top-flow rank {first.get('rank')}")
            if math.isfinite(safe_float(first.get("net_premium"))):
                parts.append(f"net premium ${safe_float(first.get('net_premium')):,.0f}")
            if _clean(first.get("flow_direction")):
                parts.append(f"flow {first.get('flow_direction')}")
    oi = _clean(row.get("oi_carryover_status"))
    if oi:
        parts.append(f"OI {oi}")
    reason = _clean(row.get("oi_carryover_reason"))
    if reason and len("; ".join(parts)) < 180:
        parts.append(reason)
    return "; ".join(parts) if parts else _clean(row.get("Expected value source"))


def _regime_fit(row: pd.Series | dict[str, Any], regime: dict[str, Any]) -> str:
    trend = _clean(regime.get("trend")) or "unknown"
    vol = _clean(regime.get("volatility")) or "unknown"
    direction = _clean(row.get("direction"))
    if trend == "uptrend" and direction in {"Bull Put", "Bull Call"}:
        fit = "aligned"
    elif trend == "downtrend" and direction in {"Bear Call", "Bear Put"}:
        fit = "aligned"
    elif trend == "range" and _is_credit(row):
        fit = "income-friendly"
    else:
        fit = "conflicted"
    return f"{fit}: trend={trend}, vol={vol}, flow={regime.get('flow', 'unknown')}"


def _catalyst_risk(row: pd.Series | dict[str, Any]) -> str:
    status = _clean(row.get("catalyst_status")) or "unknown"
    days = safe_float(row.get("catalyst_earnings_days"))
    if not math.isfinite(days):
        days = safe_float(row.get("earnings_days"))
    note = _clean(row.get("catalyst_note"))
    event_date = earnings_event_date(row)
    asof = _parse_date(
        _clean(
            row.get("overlay_evaluation_date")
            or row.get("v4_asof")
            or row.get("asof")
        )
    )
    expiry = _parse_date(_clean(row.get("expiry")))
    if event_date is not None:
        if not math.isfinite(days) and asof is not None:
            days = float((event_date - asof).days)
        relation = ""
        if expiry is not None:
            relation = "; after expiry" if event_date > expiry else "; on/before expiry"
        status_text = status if status.lower() not in {"", "unknown"} else "ticker-specific news unconfirmed"
        day_text = f" ({int(days)} days)" if math.isfinite(days) else ""
        return (
            f"{status_text}; next earnings {event_date}{day_text}{relation}"
            + (f"; {note}" if note else "")
        )
    if math.isfinite(days):
        return f"{status}; earnings in {int(days)} days" + (f"; {note}" if note else "")
    if is_etf_row(pd.Series(row)):
        return "ETF/index; no single-company earnings gate" + (f"; {note}" if note else "")
    return status + (f"; {note}" if note else "")


def _target_methodology(row: pd.Series | dict[str, Any]) -> str:
    width = safe_float(row.get("spread_width"))
    credit_pct = safe_float(row.get("credit_pct_width"))
    debit_pct = safe_float(row.get("debit_pct_width"))
    expected = safe_float(row.get("expected_move_ratio"))
    quote = safe_float(row.get("quote_width_pct"))
    sample = safe_float(row.get("edge_sample_size"))
    win = safe_float(row.get("edge_win_rate"))
    parts = []
    if _is_credit(row) and math.isfinite(credit_pct):
        parts.append(f"target credit is {credit_pct:.1%} of ${width:g} width")
    elif _is_debit(row) and math.isfinite(debit_pct):
        parts.append(f"target debit is {debit_pct:.1%} of ${width:g} width")
    target = _ticket_entry_target(row, _disposition(row, targetable=True))
    if target:
        parts.append(f"entry target {target}")
    if math.isfinite(expected):
        parts.append(f"expected-move support ratio {expected:.2f}")
    if math.isfinite(quote):
        parts.append(f"quote width {quote:.1%}")
    if math.isfinite(sample):
        win_text = f", win {win:.0%}" if math.isfinite(win) else ""
        parts.append(f"prior outcome sample n={int(sample)}{win_text}")
    flow = _clean(row.get("flow_quality"))
    oi = _clean(row.get("oi_carryover_status"))
    if flow or oi:
        parts.append(f"flow/OI quality {flow or 'unknown'}/{oi or 'unknown'}")
    return "; ".join(parts) if parts else "fresh Schwab quote and risk/reward recheck set the work limit"


def _hold_window(row: pd.Series | dict[str, Any]) -> str:
    dte = safe_float(row.get("dte"))
    if _is_debit(row):
        return "Hold only while the directional thesis remains confirmed; review at 7 DTE, but do not force an unvalidated time-stop exit."
    if math.isfinite(dte) and dte <= 10:
        return "1-5 trading days; do not carry unmanaged expiration-week gamma"
    if math.isfinite(dte) and dte <= 30:
        return "2-10 trading days; reassess at 7 DTE and exit on thesis or price invalidation"
    return "3-15 trading days; reassess if thesis or volatility regime changes"


def _dte_text(row: pd.Series | dict[str, Any]) -> str:
    dte = safe_float(row.get("dte"), safe_float(row.get("DTE")))
    if math.isfinite(dte) and dte >= 0:
        return str(int(round(dte)))
    expiry = _parse_date(_clean(row.get("expiry") or row.get("Expiry")))
    asof = _parse_date(_clean(row.get("v4_asof") or row.get("asof")))
    if expiry is not None and asof is not None and expiry >= asof:
        return str((expiry - asof).days)
    return ""


def _stop_text(row: pd.Series | dict[str, Any]) -> str:
    existing = _clean(row.get("stop_loss") or row.get("thesis_invalidation"))
    if existing:
        return existing
    max_loss = _max_loss_value(row)
    if _is_debit(row):
        debit = safe_float(row.get("debit"))
        return f"Review if spread loses 40-50% of debit" + (f" (${debit:.2f})" if math.isfinite(debit) else "")
    credit = _expectancy_safe_entry_price(row)
    if not math.isfinite(credit):
        credit = safe_float(row.get("credit"))
    if math.isfinite(credit):
        return f"Review/close if debit reaches about ${credit * 2:.2f} or short strike is threatened"
    if math.isfinite(max_loss):
        return f"Keep loss below ${max_loss:,.0f} per contract"
    return "Invalidate on thesis break, fresh hard news, or risk-cap breach"


def _gap_risk_plan(row: pd.Series | dict[str, Any], disposition: str) -> str:
    direction = _clean(row.get("direction") or row.get("Trade")).lower()
    entry = (
        _ticket_entry_target(row, disposition)
        if _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row)
        else _entry_target(row)
    )
    if disposition in {"Research", "Avoid"}:
        return "Gap +1% or -1%: keep in Research/Avoid until the hard blocker clears and the setup is repriced."
    if "bull" in direction:
        return (
            f"Gap +1%: bullish thesis improves, but do not chase; enter only if {entry} still holds with fresh OI/news confirmation. "
            "Gap -1%: thesis weakens; downgrade to Scout/Research unless support, flow, and quote width reconfirm."
        )
    if "bear" in direction:
        return (
            f"Gap -1%: bearish thesis improves, but do not chase; enter only if {entry} still holds with fresh OI/news confirmation. "
            "Gap +1%: thesis weakens; downgrade to Scout/Research unless resistance, flow, and quote width reconfirm."
        )
    return (
        f"Gap +1% or -1%: reprice the structure at the open; use {entry} as the limit and stand down if reward/risk or liquidity degrades."
    )


def _entry_price_from_target(row: pd.Series | dict[str, Any]) -> float:
    target = _entry_target(row)
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", target)
    if match:
        return safe_float(match.group(1))
    return safe_float(row.get("required_entry"), safe_float(row.get("credit") if _is_credit(row) else row.get("debit")))


def _oco_bracket_logic(row: pd.Series | dict[str, Any], disposition: str) -> str:
    medium_debit = _medium_debit_sleeve_eligible(row)
    if _payoff_evidence_ready(row) or medium_debit:
        entry = _expectancy_safe_entry_price(row)
        current = _v4_current_entry_price(row)
        if (
            _is_debit(row)
            and math.isfinite(current)
            and current > 0
            and math.isfinite(entry)
            and current <= entry
        ):
            entry = current
        target = _ticket_entry_target(row, disposition)
    else:
        entry = _entry_price_from_target(row)
        target = _entry_target(row)
    trade = _trade_legs(row)
    prefix = "Submit only with OCO attached" if disposition == "Execute" else "If this converts to Execute, attach OCO"
    if not math.isfinite(entry) or entry <= 0:
        return f"{prefix}: enter {trade} only after a fresh Schwab quote; bracket must include explicit profit-taking and stop legs before entry."
    if _is_debit(row):
        target_profit = _entry_limit_expectancy(row)[0] if medium_debit else math.nan
        take_profit = entry + target_profit / 100.0 if math.isfinite(target_profit) else entry * 1.60
        stop_value = entry * 0.50
        return (
            f"{prefix}: entry {trade} at {target}; OCO take-profit SELL TO CLOSE near ${take_profit:.2f} credit, "
            f"stop SELL TO CLOSE if spread value falls near ${stop_value:.2f} or thesis breaks."
        )
    take_profit_debit = entry * (1.0 - PROFIT_TAKE_PCT)
    return (
        f"{prefix}: entry {trade} at {target}; if filled at the minimum ${entry:.2f} credit, set OCO take-profit "
        f"BUY TO CLOSE near ${take_profit_debit:.2f} debit ({PROFIT_TAKE_PCT:.0%} captured); for a better fill, "
        f"use {1.0 - PROFIT_TAKE_PCT:.0%} of the actual opening credit. No hard stop leg: maximum loss is already defined by the "
        "spread width, and replay showed every stop-carrying exit rule underperforming its stopless twin. "
        "Manage by thesis or roll if the short strike is breached."
    )


def _safety_flags_text(row: pd.Series | dict[str, Any]) -> str:
    flags = _clean(row.get("v4_safety_flags"))
    penalties = _clean(row.get("penalties")).lower()
    if "risk_capped" in penalties and "Risk Capped" not in flags:
        flags = _append_note(flags, "Risk Capped")
    if not flags and _safety_research_reason(row):
        flags = "Safety Research"
    return flags or "None"


def _manual_instruction(row: pd.Series | dict[str, Any], disposition: str) -> str:
    if disposition == "Execute":
        return "Fresh Schwab quote, news, OI, portfolio-risk check, and OCO bracket must all pass before manual order entry."
    if disposition == "Scout":
        return "NOT AN ORDER - re-score into Work Limit or Execute after news, OI/flow, spread-width, and portfolio checks before any entry."
    if disposition == "Wheel/Cash":
        return "Confirm assignment quality, cash budget, no near-term earnings, and no duplicate exposure before selling premium."
    return "NOT AN ORDER - target only; fresh Schwab quote, news, OI/flow, and re-score required before entry."


def _why_review(row: pd.Series | dict[str, Any], top_flow: pd.DataFrame) -> str:
    pieces = []
    flow = _flow_oi_evidence(row, top_flow)
    if flow:
        pieces.append(flow)
    edge = _clean(row.get("edge_verdict") or row.get("replay_ev_verdict"))
    if edge:
        pieces.append(f"historical edge={edge}")
    if _payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row):
        expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
        if math.isfinite(expected_value) and not math.isnan(profit_factor):
            pf_text = f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf"
            authority = (
                "validated medium-debit route"
                if _medium_debit_sleeve_eligible(row)
                else "validated"
                if _payoff_model_ready(row)
                else "probationary"
            )
            pieces.append(f"{authority} final-structure EV={_money(expected_value)}, PF={pf_text}")
    else:
        pieces.append("realized payoff lane is unvalidated; displayed only for research")
    reason = _clean(
        row.get("v4_direct_disposition_reason")
        or row.get("trade_status_reason")
        or row.get("Why Execute, Scout, Research, or Avoid")
    )
    if reason:
        pieces.append(reason)
    return "; ".join(pieces)[:500]


def _ticket_row_from_scored(
    row: pd.Series,
    *,
    rank: int,
    regime: dict[str, Any],
    top_flow: pd.DataFrame,
) -> dict[str, Any]:
    disposition = _disposition(row, targetable=True)
    payoff_evidence_ready = _payoff_evidence_ready(row)
    execution_evidence_ready = payoff_evidence_ready or _medium_debit_sleeve_eligible(row)
    if execution_evidence_ready:
        target_profit, max_loss, expected_value, profit_factor = _entry_limit_expectancy(row)
    else:
        target_profit = _target_profit_value(row)
        max_loss = _max_loss_value(row)
        expected_value = math.nan
        profit_factor = math.nan
    reward_risk = target_profit / max_loss if math.isfinite(target_profit) and math.isfinite(max_loss) and max_loss > 0 else math.nan
    expected_win = _reported_win_rate(row)
    expected_win_text = f"{expected_win:.0%}" if math.isfinite(expected_win) else ""
    avg_win = target_profit if execution_evidence_ready else math.nan
    avg_loss = -max_loss if execution_evidence_ready and math.isfinite(max_loss) else math.nan
    if _medium_debit_sleeve_eligible(row):
        _, _, _, _, stressed_win, stressed_loss = _medium_debit_ticket_economics(row)
        if math.isfinite(stressed_win) and math.isfinite(stressed_loss):
            avg_win = stressed_win
            avg_loss = -stressed_loss
    if _walk_forward_credit_execution_ready(row) and math.isfinite(max_loss):
        walk_forward = _walk_forward_credit_expectancy(row, max_loss)
        if math.isfinite(walk_forward[0]):
            avg_win = walk_forward[2]
            avg_loss = -walk_forward[3]
    maximum_profit = _maximum_profit_at_entry_value(row, disposition)
    return {
        "rank": rank,
        "display status": _status_label(disposition),
        "lane": _lane(row, disposition),
        "ticker": _clean(row.get("ticker")).upper(),
        "trade legs": _trade_legs(row),
        "expiry": _clean(row.get("expiry")),
        "DTE": _dte_text(row),
        "next-session swing entry target": _ticket_entry_target(row, disposition),
        "current Schwab mid/natural reference": _mid_natural(row),
        "maximum profit": _money(maximum_profit),
        "profit target": _money(target_profit),
        "max loss": _money(max_loss),
        "one-contract risk": _money(max_loss),
        "reward/risk": f"{reward_risk:.2f}" if math.isfinite(reward_risk) else "",
        "suggested size": _suggested_size(row, disposition),
        "swing hold window": _hold_window(row),
        "stop/invalidation": _stop_text(row),
        "gap-risk plan +/-1% open": _gap_risk_plan(row, disposition),
        "EOD trend evidence": _trend_evidence(row),
        "flow/OI evidence": _flow_oi_evidence(row, top_flow),
        "liquidity context": "; ".join(
            part
            for part in [
                _clean(row.get("liquidity_quality")),
                _clean(row.get("liquidity_summary")),
            ]
            if part
        ),
        "portfolio context": _unique_semicolon_context(
            row.get("portfolio_fit"),
            row.get("portfolio_warning"),
            row.get("portfolio_note"),
        ) or "no concentration flag recorded; refresh portfolio before entry",
        "regime fit": _regime_fit(row, regime),
        "catalyst/earnings risk": _catalyst_risk(row),
        "target price methodology": _target_methodology(row),
        "OCO bracket order logic": _oco_bracket_logic(row, disposition),
        "blocker before entry": _blocker_text(row),
        "manual review instruction": _manual_instruction(row, disposition),
        "why this is worth reviewing tomorrow": _why_review(row, top_flow),
        "final disposition": disposition,
        "setup family": _setup_family(row),
        "setup family key": _setup_family_key(row),
        "safety calibration flags": _safety_flags_text(row),
        "expected win rate": expected_win_text,
        "win-rate basis": _reported_win_rate_basis(row),
        "confidence evidence": _confidence_evidence_text(row),
        "confidence rating": _confidence_rating(row),
        "per-ticket replay edge": _edge_evidence_text(row),
        "payoff evidence": _payoff_evidence_text(row),
        "expected average win": _money(avg_win) if execution_evidence_ready else "UNVALIDATED",
        "expected average loss": _money(avg_loss) if execution_evidence_ready else "UNVALIDATED",
        "expected value": _money(expected_value) if execution_evidence_ready else "UNVALIDATED",
        "implied profit factor": (
            f"{profit_factor:.2f}" if execution_evidence_ready and math.isfinite(profit_factor)
            else "inf" if execution_evidence_ready and profit_factor == math.inf
            else "UNVALIDATED"
        ),
        "_confidence_sort": expected_win if math.isfinite(expected_win) else -1.0,
        "_rank_score": _ticket_rank(row, disposition),
    }


def _ticket_row_from_board(
    row: pd.Series,
    *,
    rank: int,
    regime: dict[str, Any],
) -> dict[str, Any]:
    disposition = _disposition(row, targetable=True)
    target_profit = _target_profit_value(row)
    max_loss = _max_loss_value(row)
    maximum_profit = _maximum_profit_at_entry_value(row, disposition)
    reward_risk = target_profit / max_loss if math.isfinite(target_profit) and math.isfinite(max_loss) and max_loss > 0 else math.nan
    return {
        "rank": rank,
        "lane": _clean(row.get("Lane")) or _lane(row, disposition),
        "ticker": _clean(row.get("Ticker")).upper(),
        "trade legs": _trade_legs(row),
        "expiry": _clean(row.get("Expiry")),
        "DTE": _dte_text(row),
        "next-session swing entry target": _entry_target(row),
        "current Schwab mid/natural reference": _mid_natural(row),
        "maximum profit": _money(maximum_profit),
        "profit target": _money(target_profit),
        "max loss": _money(max_loss),
        "one-contract risk": _money(max_loss),
        "reward/risk": f"{reward_risk:.2f}" if math.isfinite(reward_risk) else "",
        "suggested size": "1 contract / cash secured only after assignment-quality review",
        "swing hold window": _hold_window(row),
        "stop/invalidation": _stop_text(row),
        "gap-risk plan +/-1% open": _gap_risk_plan(row, disposition),
        "EOD trend evidence": _clean(row.get("EOD trend evidence")),
        "flow/OI evidence": _clean(row.get("Expected value source")),
        "liquidity context": _clean(row.get("Liquidity")) or "manual liquidity recheck required",
        "portfolio context": "manual assignment and portfolio-overlap review required",
        "regime fit": _regime_fit(row, regime),
        "catalyst/earnings risk": "manual assignment/news check required" if disposition == "Wheel/Cash" else "manual catalyst check required",
        "target price methodology": _target_methodology(row),
        "OCO bracket order logic": _oco_bracket_logic(row, disposition),
        "blocker before entry": _clean(row.get("Required confirmation")),
        "manual review instruction": _manual_instruction(row, disposition),
        "why this is worth reviewing tomorrow": _clean(row.get("Why Execute, Scout, Research, or Avoid")),
        "display status": _status_label(disposition),
        "final disposition": disposition,
        "setup family": _setup_family(row),
        "setup family key": _setup_family_key(row),
        "safety calibration flags": _safety_flags_text(row),
        "expected win rate": "",
        "win-rate basis": "",
        "confidence evidence": "",
        "confidence rating": "⚪ UNRATED",
        "per-ticket replay edge": _edge_evidence_text(row),
        "payoff evidence": "",
        "expected average win": "",
        "expected average loss": "",
        "_confidence_sort": -1.0,
        "_rank_score": 40.0 if disposition == "Wheel/Cash" else 30.0,
    }


def _lane(row: pd.Series | dict[str, Any], disposition: str) -> str:
    ticker = _clean(row.get("ticker") or row.get("Ticker")).upper()
    strategy = f"{row.get('strategy', '')} {row.get('Trade', '')}".lower()
    if disposition == "Wheel/Cash" or "cash-secured" in strategy or "wheel" in strategy:
        return "Wheel/Cash"
    if ticker in INDEX_FLOW_TICKERS or bool(row.get("index_fallback", False)):
        return "Index/ETF"
    if _is_debit(row):
        return "Momentum Debit"
    if disposition == "Scout":
        return "Scout"
    return "Swing Target / Work Limit" if disposition == "Swing Target / Work Limit" else disposition


def _suggested_size(row: pd.Series | dict[str, Any], disposition: str) -> str:
    contracts = safe_float(row.get("contracts"))
    if disposition == "Execute":
        if "pilot" in _clean(row.get("trade_tier")).lower():
            return "1 contract only; probationary evidence cap"
        if _is_debit(row) and _clean(row.get("debit_policy_tier")).lower() == "medium":
            return "1 contract only; Medium debit sleeve"
        if _is_credit(row) and _clean(row.get("credit_policy_tier")).lower() == "medium":
            return "1 contract only; Medium credit sleeve"
        qty = int(contracts) if math.isfinite(contracts) and contracts > 0 else 1
        return f"{qty} contract{'s' if qty != 1 else ''}; only if all live checks still pass"
    if disposition == "Scout":
        return "1 contract only"
    if disposition == "Wheel/Cash":
        return "1 contract only if cash-secured and assignment-quality review passes"
    return "1 contract review; size only after fresh target fill and risk-cap check"


def _ticket_rank(row: pd.Series | dict[str, Any], disposition: str) -> float:
    rank = {"Execute": 100.0, "Swing Target / Work Limit": 80.0, "Scout": 65.0, "Wheel/Cash": 55.0}.get(disposition, 20.0)
    rank += safe_float(row.get("confirmation_score"), 0.0)
    rank += safe_float(row.get("score"), 0.0)
    target = _target_profit_value(row)
    risk = _max_loss_value(row)
    if math.isfinite(target):
        rank += min(10.0, target / 50.0)
    if math.isfinite(target) and math.isfinite(risk) and risk > 0:
        rank += min(5.0, (target / risk) * 10.0)
    sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size")))
    expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
    confidence = _effective_win_rate(row)
    if math.isfinite(confidence):
        rank += max(0.0, min(10.0, confidence * 10.0))
    if math.isfinite(sample) and sample >= V4_MIN_REPLAY_SAMPLE:
        if math.isfinite(expected_value) and math.isfinite(risk) and risk > 0:
            rank += max(-10.0, min(10.0, (expected_value / risk) * 20.0))
        if math.isfinite(profit_factor):
            rank += max(-5.0, min(5.0, (profit_factor - 1.0) * 5.0))
    else:
        rank -= 15.0
    return rank


def build_v4_swing_target_tickets(
    *,
    scored: pd.DataFrame,
    board: pd.DataFrame,
    regime: dict[str, Any],
    top_flow: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scored is not None and not scored.empty:
        for _, row in scored.iterrows():
            if _targetable(row):
                rows.append(_ticket_row_from_scored(row, rank=0, regime=regime, top_flow=top_flow))
    # Opportunity-board Wheel/Cash rows are contextual research until a scored
    # structure carries complete execution evidence, exact option-leg pricing,
    # confidence, and payoff fields.  A dollar-denominated review trigger alone
    # must never promote a board row into the actionable ticket book.
    if not rows:
        return pd.DataFrame(columns=V4_TARGET_TICKET_COLUMNS)
    tickets = pd.DataFrame(rows)
    tickets["_disposition_sort"] = tickets["final disposition"].map(
        {"Execute": 3, "Swing Target / Work Limit": 2, "Scout": 1, "Wheel/Cash": 0}
    ).fillna(-1)
    tickets["_dedupe"] = (
        tickets["ticker"].astype(str)
        + "|"
        + tickets["trade legs"].astype(str)
        + "|"
        + tickets["expiry"].astype(str)
    )
    tickets = (
        tickets.sort_values(
            ["_disposition_sort", "_confidence_sort", "_rank_score"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .drop_duplicates("_dedupe", keep="first")
        .drop(columns=["_dedupe", "_disposition_sort"])
        .reset_index(drop=True)
    )
    tickets["rank"] = range(1, len(tickets) + 1)
    for col in V4_TARGET_TICKET_COLUMNS:
        if col not in tickets.columns:
            tickets[col] = ""
    return tickets[V4_TARGET_TICKET_COLUMNS + ["_rank_score"]].drop(columns=["_rank_score"])


def _ticket_contract_qty(value: object) -> int:
    match = re.search(r"(\d+)\s+contract", _clean(value).lower())
    if not match:
        return 1
    return max(1, int(safe_float(match.group(1), 1.0)))


def apply_v4_risk_cap(tickets: pd.DataFrame, portfolio: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_columns = [
        "ticker",
        "trade legs",
        "final disposition before",
        "final disposition after",
        "account_value",
        "two_percent_loss_cap",
        "max_loss_before",
        "max_loss_after",
        "size_before",
        "size_after",
        "risk_cap_action",
        "risk_capped",
    ]
    if tickets is None or tickets.empty:
        return tickets.copy() if tickets is not None else pd.DataFrame(), pd.DataFrame(columns=audit_columns)
    account_value = safe_float((portfolio or {}).get("total_value"))
    if not math.isfinite(account_value) or account_value <= 0:
        rows = []
        for _, row in tickets.iterrows():
            rows.append(
                {
                    "ticker": row.get("ticker"),
                    "trade legs": row.get("trade legs"),
                    "final disposition before": row.get("final disposition"),
                    "final disposition after": row.get("final disposition"),
                    "account_value": math.nan,
                    "two_percent_loss_cap": math.nan,
                    "max_loss_before": _money_value(row.get("max loss")),
                    "max_loss_after": _money_value(row.get("max loss")),
                    "size_before": row.get("suggested size"),
                    "size_after": row.get("suggested size"),
                    "risk_cap_action": "account_value_unavailable",
                    "risk_capped": False,
                }
            )
        return tickets.copy(), pd.DataFrame(rows, columns=audit_columns)
    cap = account_value * 0.02
    out = tickets.copy()
    rows: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        before_disp = _clean(row.get("final disposition"))
        max_loss_before = _money_value(row.get("max loss"))
        size_before = _clean(row.get("suggested size"))
        # Advisory only: a dollar cap must never downgrade a disposition or resize a ticket.
        over_cap = (
            before_disp in {"Execute", "Swing Target / Work Limit"}
            and math.isfinite(max_loss_before)
            and max_loss_before > cap
        )
        if over_cap:
            existing_flags = row.get("safety calibration flags")
            if _clean(existing_flags).lower() == "none":
                existing_flags = ""
            out.at[idx, "safety calibration flags"] = _append_note(
                existing_flags,
                f"Sizing note: max loss ${max_loss_before:,.0f} exceeds 2% of account (${cap:,.0f})",
            )
        rows.append(
            {
                "ticker": row.get("ticker"),
                "trade legs": row.get("trade legs"),
                "final disposition before": before_disp,
                "final disposition after": before_disp,
                "account_value": round(account_value, 2),
                "two_percent_loss_cap": round(cap, 2),
                "max_loss_before": round(max_loss_before, 2) if math.isfinite(max_loss_before) else math.nan,
                "max_loss_after": round(max_loss_before, 2) if math.isfinite(max_loss_before) else math.nan,
                "size_before": size_before,
                "size_after": size_before,
                "risk_cap_action": "advisory_note_only" if over_cap else "within_cap",
                "risk_capped": False,
            }
        )
    return out, pd.DataFrame(rows, columns=audit_columns)


def build_raw_universe(
    *,
    top_flow: pd.DataFrame,
    scored: pd.DataFrame,
    candidates: pd.DataFrame,
    portfolio: dict[str, Any],
) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}

    def add(ticker: object, source: str, **extra: Any) -> None:
        symbol = _clean(ticker).upper()
        if not symbol:
            return
        row = rows.setdefault(symbol, {"ticker": symbol, "discovery_sources": set()})
        row["discovery_sources"].add(source)
        row.update({k: v for k, v in extra.items() if v not in {None, ""}})

    for ticker in sorted(FIXED_LIQUID_UNIVERSE):
        add(ticker, "fixed liquid universe")
    for ticker in sorted(INDEX_FLOW_TICKERS):
        add(ticker, "index/ETF universe")
    for ticker in sorted(set(SECTOR_BENCHMARKS.values())):
        add(ticker, "sector ETF proxy")
    for ticker in (portfolio or {}).get("equity_exposure", {}):
        add(ticker, "current Schwab portfolio name", portfolio_exposure=(portfolio or {}).get("equity_exposure", {}).get(ticker))
    for ticker in (portfolio or {}).get("option_underlyings", []):
        add(ticker, "current Schwab option underlying")
    if top_flow is not None and not top_flow.empty:
        for _, row in top_flow.iterrows():
            add(
                row.get("ticker"),
                "top 50 UW net premium / flow velocity",
                top_flow_rank=row.get("rank"),
                top_flow_net_premium=row.get("net_premium"),
                top_flow_direction=row.get("flow_direction"),
                top_flow_source=row.get("source"),
                top_flow_rank_score=row.get("rank_score"),
            )
    scored_tickers = set(scored.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper()) if scored is not None and not scored.empty else set()
    candidate_tickers = set(candidates.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper()) if candidates is not None and not candidates.empty else set()
    out_rows = []
    for ticker, row in sorted(rows.items()):
        row = row.copy()
        sources = sorted(row.pop("discovery_sources"))
        row["discovered"] = True
        row["discovery_sources"] = "; ".join(sources)
        row["candidate_generated"] = ticker in candidate_tickers or ticker in scored_tickers
        row["scored_construction_count"] = int((scored.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().eq(ticker)).sum()) if scored is not None and not scored.empty else 0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _best_scored_by_ticker(scored: pd.DataFrame) -> dict[str, pd.Series]:
    if scored is None or scored.empty or "ticker" not in scored.columns:
        return {}
    rows = []
    for idx, row in scored.iterrows():
        targetable = _targetable(row)
        disposition = _disposition(row, targetable=targetable)
        rows.append((idx, _ticket_rank(row, disposition)))
    rank = pd.Series({idx: score for idx, score in rows})
    work = scored.copy()
    work["_v4_rank"] = rank
    return {
        ticker: part.sort_values("_v4_rank", ascending=False).iloc[0].drop(labels=["_v4_rank"], errors="ignore")
        for ticker, part in work.groupby(work["ticker"].astype(str).str.upper())
    }


def _target_key_set(tickets: pd.DataFrame) -> set[tuple[str, str, str]]:
    if tickets.empty:
        return set()
    return {
        (_clean(row.get("ticker")).upper(), _clean(row.get("trade legs")), _clean(row.get("expiry")))
        for _, row in tickets.iterrows()
    }


def _row_target_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    return (_clean(row.get("ticker") or row.get("Ticker")).upper(), _trade_legs(row), _clean(row.get("expiry") or row.get("Expiry")))


def build_candidate_disposition(
    *,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    top_flow: pd.DataFrame,
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    target_keys = _target_key_set(tickets)
    rows: list[dict[str, Any]] = []
    if scored is not None and not scored.empty:
        grouped = scored.groupby(scored["ticker"].astype(str).str.upper())
        construction_counts = grouped.size().to_dict()
        for _, row in scored.iterrows():
            targetable = _targetable(row)
            disposition = _disposition(row, targetable=targetable)
            hard_reason = _hard_blocker_reason(row)
            safety_reason = _safety_research_reason(row)
            key = _row_target_key(row)
            rows.append(
                {
                    "ticker": _clean(row.get("ticker")).upper(),
                    "thesis": _clean(row.get("direction")),
                    "strategy": _clean(row.get("strategy")),
                    "expiry": _clean(row.get("expiry")),
                    "human_readable_trade_legs": _trade_legs(row),
                    "discovered": True,
                    "candidate_generated": True,
                    "constructions_attempted": int(construction_counts.get(_clean(row.get("ticker")).upper(), 0)),
                    "best_construction": _trade_legs(row),
                    "final_disposition": disposition,
                    "target_ticket_shown": key in target_keys,
                    "targetable": targetable,
                    "exact_reason_if_not_targetable": "" if targetable else hard_reason or safety_reason or _blocker_text(row),
                    "hard_or_non_hard_blocker": "hard" if hard_reason else "safety" if safety_reason else "non-hard",
                    "current_reference": _mid_natural(row),
                    "target_entry": _entry_target(row),
                    "price_target_miss_visible": "below target" in _clean(row.get("price_annotation")).lower() or "above target" in _clean(row.get("price_annotation")).lower(),
                }
            )
    candidate_tickers = set(candidates.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper()) if candidates is not None and not candidates.empty else set()
    scored_tickers = set(scored.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper()) if scored is not None and not scored.empty else set()
    missing_scored = sorted(candidate_tickers - scored_tickers)
    for ticker in missing_scored:
        rows.append(
            {
                "ticker": ticker,
                "thesis": "",
                "strategy": "",
                "expiry": "",
                "human_readable_trade_legs": "",
                "discovered": True,
                "candidate_generated": True,
                "constructions_attempted": 0,
                "best_construction": "",
                "final_disposition": "Research",
                "target_ticket_shown": False,
                "targetable": False,
                "exact_reason_if_not_targetable": "candidate generated but Schwab scoring returned no usable construction",
                "hard_or_non_hard_blocker": "hard",
                "current_reference": "",
                "target_entry": "",
                "price_target_miss_visible": False,
            }
        )
    if top_flow is not None and not top_flow.empty:
        generated = set(candidate_tickers) | scored_tickers
        for _, flow in top_flow.iterrows():
            ticker = _clean(flow.get("ticker")).upper()
            if ticker and ticker not in generated:
                rows.append(
                    {
                        "ticker": ticker,
                        "thesis": _clean(flow.get("flow_direction")),
                        "strategy": "",
                        "expiry": "",
                        "human_readable_trade_legs": "",
                        "discovered": True,
                        "candidate_generated": False,
                        "constructions_attempted": 0,
                        "best_construction": "",
                        "final_disposition": "Research",
                        "target_ticket_shown": False,
                        "targetable": False,
                        "exact_reason_if_not_targetable": "top-flow ticker discovered but no candidate generated; likely missing usable hot-chain contract, expiry/strike window, or options-liquidity screen",
                        "hard_or_non_hard_blocker": "non-hard",
                        "current_reference": "",
                        "target_entry": "",
                        "price_target_miss_visible": False,
                    }
                )
    return pd.DataFrame(rows)


def build_construction_attempts(
    *,
    scored: pd.DataFrame,
    top_flow: pd.DataFrame,
    tickets: pd.DataFrame,
    portfolio: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scored is not None and not scored.empty:
        group = scored.groupby(scored["ticker"].astype(str).str.upper())
        expiry_counts = group["expiry"].nunique().to_dict() if "expiry" in scored.columns else {}
        strike_counts = group["short_strike"].nunique().to_dict() if "short_strike" in scored.columns else {}
        strategy_sets = group["strategy"].agg(lambda values: sorted(set(_clean(v) for v in values if _clean(v)))).to_dict()
        wheel_tickers = set(tickets[tickets["final disposition"].eq("Wheel/Cash")]["ticker"].astype(str).str.upper()) if not tickets.empty else set()
        for idx, row in scored.iterrows():
            ticker = _clean(row.get("ticker")).upper()
            targetable = _targetable(row)
            disposition = _disposition(row, targetable=targetable)
            safety_reason = _safety_research_reason(row)
            rows.append(
                {
                    "attempt_id": idx + 1,
                    "ticker": ticker,
                    "thesis": _clean(row.get("direction")),
                    "strategy_attempted": _clean(row.get("strategy")),
                    "construction_source": _clean(row.get("construction_source")),
                    "live_construction_source": _clean(row.get("live_construction_source")),
                    "expiry": _clean(row.get("expiry")),
                    "human_readable_trade_legs": _trade_legs(row),
                    "alternate_strike_attempted_for_ticker": int(strike_counts.get(ticker, 0)) > 1,
                    "alternate_expiry_attempted_for_ticker": int(expiry_counts.get(ticker, 0)) > 1,
                    "credit_alternative_attempted": any("Credit" in item for item in strategy_sets.get(ticker, [])),
                    "debit_alternative_attempted": any("Debit" in item for item in strategy_sets.get(ticker, [])),
                    "calendar_or_diagonal_attempted": False,
                    "ETF_or_index_proxy_attempted": ticker in INDEX_FLOW_TICKERS,
                    "wheel_cash_version_attempted": ticker in wheel_tickers,
                    "smaller_defined_risk_version_attempted": True,
                    "current_reference": _mid_natural(row),
                    "target_entry": _entry_target(row),
                    "target_price_methodology": _target_methodology(row),
                    "final_disposition": disposition,
                    "exact_result_reason": safety_reason or _blocker_text(row),
                    "hard_or_non_hard_blocker": "hard" if _is_hard_blocked(row) else "safety" if safety_reason else "non-hard",
                }
            )
    generated_tickers = set(scored.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper()) if scored is not None and not scored.empty else set()
    if top_flow is not None and not top_flow.empty:
        for _, row in top_flow.iterrows():
            ticker = _clean(row.get("ticker")).upper()
            if ticker and ticker not in generated_tickers:
                rows.append(
                    {
                        "attempt_id": len(rows) + 1,
                        "ticker": ticker,
                        "thesis": _clean(row.get("flow_direction")),
                        "strategy_attempted": "",
                        "construction_source": "top_flow_discovery",
                        "live_construction_source": "",
                        "expiry": "",
                        "human_readable_trade_legs": "",
                        "alternate_strike_attempted_for_ticker": False,
                        "alternate_expiry_attempted_for_ticker": False,
                        "credit_alternative_attempted": False,
                        "debit_alternative_attempted": False,
                        "calendar_or_diagonal_attempted": False,
                        "ETF_or_index_proxy_attempted": ticker in INDEX_FLOW_TICKERS,
                        "wheel_cash_version_attempted": False,
                        "smaller_defined_risk_version_attempted": False,
                        "current_reference": "",
                        "target_entry": "",
                        "target_price_methodology": "",
                        "final_disposition": "Research",
                        "exact_result_reason": "top-flow ticker discovered but no construction was generated by available EOD chain windows",
                        "hard_or_non_hard_blocker": "non-hard",
                    }
                )
    return pd.DataFrame(rows)


def build_suppression_audit(dispositions: pd.DataFrame) -> pd.DataFrame:
    if dispositions.empty:
        return pd.DataFrame()
    suppressed = dispositions[~dispositions["target_ticket_shown"].astype(bool)].copy()
    rows = []
    for _, row in suppressed.iterrows():
        reason = _clean(row.get("exact_reason_if_not_targetable"))
        if not reason:
            reason = "lower-ranked duplicate construction or Research-quality candidate stayed visible in disposition audit"
        price_miss_hidden = bool(row.get("price_target_miss_visible")) and bool(row.get("targetable"))
        rows.append(
            {
                "ticker": row.get("ticker"),
                "thesis": row.get("thesis"),
                "strategy": row.get("strategy"),
                "human_readable_trade_legs": row.get("human_readable_trade_legs"),
                "final_disposition": row.get("final_disposition"),
                "suppressed_from_target_tickets": True,
                "exact_reason": reason,
                "hard_or_non_hard_blocker": row.get("hard_or_non_hard_blocker"),
                "targetable_trade_hidden_by_price_miss": price_miss_hidden,
            }
        )
    return pd.DataFrame(rows)


def build_no_miss_audit(
    *,
    top_flow: pd.DataFrame,
    scored: pd.DataFrame,
    dispositions: pd.DataFrame,
    attempts: pd.DataFrame,
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best = _best_scored_by_ticker(scored)
    ticket_tickers = set(tickets.get("ticker", pd.Series(dtype=object)).astype(str).str.upper()) if tickets is not None and not tickets.empty else set()
    disposition_by_ticker = dispositions.groupby("ticker") if dispositions is not None and not dispositions.empty else None
    if attempts is not None and not attempts.empty:
        real_attempts = attempts[
            attempts.get("strategy_attempted", pd.Series("", index=attempts.index)).astype(str).str.len().gt(0)
            | attempts.get("human_readable_trade_legs", pd.Series("", index=attempts.index)).astype(str).str.len().gt(0)
        ].copy()
        attempt_counts = real_attempts.groupby("ticker").size().to_dict() if not real_attempts.empty else {}
    else:
        attempt_counts = {}
    if top_flow is None or top_flow.empty:
        return pd.DataFrame(columns=["ticker", "discovered", "candidate_generated", "constructions_attempted", "best_construction", "final_disposition"])
    for _, flow in top_flow.iterrows():
        ticker = _clean(flow.get("ticker")).upper()
        best_row = best.get(ticker)
        candidate_generated = best_row is not None
        if best_row is not None:
            disposition = _disposition(best_row, targetable=_targetable(best_row))
            best_construction = _trade_legs(best_row)
            safety_reason = _safety_research_reason(best_row)
            reason = "" if ticker in ticket_tickers else _hard_blocker_reason(best_row) or safety_reason or _blocker_text(best_row)
            blocker_type = "hard" if _is_hard_blocked(best_row) else "safety" if safety_reason else "non-hard"
        elif disposition_by_ticker is not None and ticker in disposition_by_ticker.groups:
            part = disposition_by_ticker.get_group(ticker)
            disposition = _clean(part.iloc[0].get("final_disposition"))
            best_construction = _clean(part.iloc[0].get("best_construction"))
            reason = _clean(part.iloc[0].get("exact_reason_if_not_targetable"))
            blocker_type = _clean(part.iloc[0].get("hard_or_non_hard_blocker"))
        else:
            disposition = "Research"
            best_construction = ""
            reason = "top-flow ticker discovered but no candidate generated by EOD construction windows"
            blocker_type = "non-hard"
        rows.append(
            {
                "top_flow_rank": flow.get("rank"),
                "ticker": ticker,
                "discovered": True,
                "candidate_generated": candidate_generated,
                "constructions_attempted": int(attempt_counts.get(ticker, 0)),
                "best_construction": best_construction,
                "final_disposition": disposition,
                "shown_as_target_ticket": ticker in ticket_tickers,
                "if_not_targetable_exact_reason": "" if ticker in ticket_tickers else reason,
                "hard_or_non_hard_blocker": "" if ticker in ticket_tickers else blocker_type,
                "flow_direction": flow.get("flow_direction"),
                "net_premium": flow.get("net_premium"),
                "rank_score": flow.get("rank_score"),
            }
        )
    return pd.DataFrame(rows)


def build_generation_audit(
    *,
    top_flow: pd.DataFrame,
    scored: pd.DataFrame,
    dispositions: pd.DataFrame,
    tickets: pd.DataFrame,
    board: pd.DataFrame,
    secondary_sweep: pd.DataFrame | None = None,
) -> pd.DataFrame:
    research_ratio = 0.0
    if dispositions is not None and not dispositions.empty:
        research_ratio = float(dispositions["final_disposition"].eq("Research").mean())
    strategies = set(scored.get("strategy", pd.Series(dtype=object)).dropna().astype(str)) if scored is not None and not scored.empty else set()
    target_hidden_by_price = False
    if dispositions is not None and not dispositions.empty:
        missing_targetable = dispositions[(dispositions["targetable"].astype(bool)) & (~dispositions["target_ticket_shown"].astype(bool))]
        target_hidden_by_price = bool(missing_targetable["price_target_miss_visible"].astype(bool).any()) if not missing_targetable.empty else False
    secondary_triggered = False
    if secondary_sweep is not None and not secondary_sweep.empty and "triggered" in secondary_sweep.columns:
        secondary_triggered = bool(secondary_sweep["triggered"].astype(bool).any())
    evidence_integrity = build_execution_evidence_integrity(scored)
    issue_rows = [
        ("expiry window too narrow", "warn" if scored is not None and not scored.empty and scored["expiry"].nunique() < 2 else "ok", "candidate expiries should span more than one usable date"),
        ("strike window too narrow", "warn" if scored is not None and not scored.empty and scored.get("short_strike", pd.Series()).nunique() < 4 else "ok", "construction attempts should include alternate strikes"),
        ("too much whale dependence", "warn" if tickets is not None and not tickets.empty and tickets["flow/OI evidence"].astype(str).str.len().mean() > 220 else "ok", "manual confirmation remains required for flow-only theses"),
        ("top-flow universe too small", "warn" if top_flow is None or len(top_flow) < 50 else "ok", f"top-flow rows={0 if top_flow is None else len(top_flow)}"),
        ("secondary liquidity sweep", "ok" if secondary_triggered or (scored is not None and len(scored) >= 3) else "warn", "triggered because candidate count fell below 3" if secondary_triggered else f"not triggered; scored rows={0 if scored is None else len(scored)}"),
        ("fixed-regime bias", "warn" if scored is not None and not scored.empty and scored["direction"].nunique() < 2 else "ok", "both bullish and bearish structures should be considered when data supports them"),
        ("too much Research classification", "warn" if research_ratio > 0.60 else "ok", f"Research ratio={research_ratio:.0%}"),
        ("missing debit alternatives", "warn" if not any("Debit" in s for s in strategies) else "ok", "debit alternatives present" if any("Debit" in s for s in strategies) else "no debit structures generated"),
        ("missing credit alternatives", "warn" if not any("Credit" in s for s in strategies) else "ok", "credit alternatives present" if any("Credit" in s for s in strategies) else "no credit structures generated"),
        ("missing index/ETF proxies", "warn" if tickets is None or tickets.empty or not tickets["lane"].eq("Index/ETF").any() else "ok", "Index/ETF lane visible"),
        ("missing wheel/cash alternatives", "warn" if board is None or board.empty or not board["Lane"].astype(str).eq("Wheel/Cash").any() else "ok", "Wheel/Cash lane visible"),
        ("overly strict confirmation gates", "warn" if tickets is not None and not tickets.empty and not tickets["final disposition"].eq("Execute").any() else "ok", "zero Execute requires target tickets and no-miss audit"),
        (
            "execution evidence reachability",
            "ok" if evidence_integrity["status"] == "PASS" else "fail",
            f"specific max n={evidence_integrity['maximum_specific_sample']} vs required n={evidence_integrity['minimum_specific_sample']}; "
            f"validated family rows={evidence_integrity['validated_family_rows']}; reachable rows={evidence_integrity['evidence_reachable_rows']}",
        ),
        ("targetable trades hidden by price miss", "fail" if target_hidden_by_price else "ok", "price target miss must remain visible as Work Limit"),
    ]
    return pd.DataFrame([{"audit_item": item, "audit_status": status, "evidence": evidence} for item, status, evidence in issue_rows])


def build_secondary_liquidity_sweep(
    *,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    top_flow: pd.DataFrame,
    flow_velocity: pd.DataFrame,
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "triggered",
        "candidate_count",
        "ticker",
        "sweep_rank",
        "relaxed_uw_block_size_filters",
        "flow_velocity_scan",
        "rolling_5m_premium",
        "vol_to_open_interest_ratio",
        "volume_oi_liquidity_ok",
        "sector_sweep",
        "spy_qqq_correlation",
        "beta_noise_ignored",
        "final_sweep_action",
        "evidence",
    ]
    candidate_count = int(len(candidates)) if candidates is not None and not candidates.empty else int(len(scored)) if scored is not None and not scored.empty else 0
    triggered = candidate_count < 3
    if not triggered:
        return pd.DataFrame(
            [
                {
                    "triggered": False,
                    "candidate_count": candidate_count,
                    "ticker": "",
                    "sweep_rank": "",
                    "relaxed_uw_block_size_filters": False,
                    "flow_velocity_scan": "not_triggered",
                    "rolling_5m_premium": math.nan,
                    "vol_to_open_interest_ratio": math.nan,
                    "volume_oi_liquidity_ok": False,
                    "sector_sweep": "not_triggered",
                    "spy_qqq_correlation": math.nan,
                    "beta_noise_ignored": False,
                    "final_sweep_action": "not_triggered_candidate_count_at_or_above_3",
                    "evidence": f"candidate_count={candidate_count}",
                }
            ],
            columns=columns,
        )
    if top_flow is None or top_flow.empty:
        return pd.DataFrame(
            [
                {
                    "triggered": True,
                    "candidate_count": candidate_count,
                    "ticker": "",
                    "sweep_rank": "",
                    "relaxed_uw_block_size_filters": True,
                    "flow_velocity_scan": "unavailable",
                    "rolling_5m_premium": math.nan,
                    "vol_to_open_interest_ratio": math.nan,
                    "volume_oi_liquidity_ok": False,
                    "sector_sweep": "unavailable",
                    "spy_qqq_correlation": math.nan,
                    "beta_noise_ignored": False,
                    "final_sweep_action": "Research",
                    "evidence": "candidate_count below 3 but top-flow universe unavailable",
                }
            ],
            columns=columns,
        )
    velocity_by_ticker = pd.DataFrame()
    if flow_velocity is not None and not flow_velocity.empty and "ticker" in flow_velocity.columns:
        velocity_by_ticker = flow_velocity.groupby(flow_velocity["ticker"].astype(str).str.upper(), as_index=True).agg(
            rolling_5m_premium=("rolling_5m_premium", "max"),
            rolling_15m_premium=("rolling_15m_premium", "max"),
            flow_velocity_signal=("flow_velocity_signal", "max"),
        )
    corr_by_ticker = pd.DataFrame()
    if correlation is not None and not correlation.empty and "ticker" in correlation.columns:
        corr_by_ticker = correlation.groupby(correlation["ticker"].astype(str).str.upper(), as_index=True).agg(
            rolling_correlation=("rolling_correlation", "first"),
            benchmark=("benchmark", "first"),
            reason=("reason", "first"),
        )
    rows: list[dict[str, Any]] = []
    for _, row in top_flow.head(50).iterrows():
        ticker = _clean(row.get("ticker")).upper()
        velocity_row = velocity_by_ticker.loc[ticker] if not velocity_by_ticker.empty and ticker in velocity_by_ticker.index else pd.Series(dtype=object)
        corr_row = corr_by_ticker.loc[ticker] if not corr_by_ticker.empty and ticker in corr_by_ticker.index else pd.Series(dtype=object)
        rolling_5m = safe_float(velocity_row.get("rolling_5m_premium"), safe_float(row.get("max_rolling_5m_premium")))
        volume_oi = safe_float(row.get("volume_oi_ratio"))
        corr = safe_float(corr_row.get("rolling_correlation"))
        beta_noise = math.isfinite(corr) and abs(corr) > 0.80
        liquidity_ok = math.isfinite(volume_oi) and volume_oi > 0
        velocity_signal = bool(velocity_row.get("flow_velocity_signal", False)) if not velocity_row.empty else math.isfinite(rolling_5m) and rolling_5m > 0
        action = "Research"
        if liquidity_ok and velocity_signal and not beta_noise:
            action = "Research - promote to manual construction queue"
        elif beta_noise:
            action = "Research - ignored as beta-noise correlation >0.8"
        rows.append(
            {
                "triggered": True,
                "candidate_count": candidate_count,
                "ticker": ticker,
                "sweep_rank": row.get("rank"),
                "relaxed_uw_block_size_filters": True,
                "flow_velocity_scan": "pass" if velocity_signal else "no_velocity_signal",
                "rolling_5m_premium": rolling_5m,
                "vol_to_open_interest_ratio": volume_oi,
                "volume_oi_liquidity_ok": bool(liquidity_ok),
                "sector_sweep": "top_50_net_flow_ex_beta_noise",
                "spy_qqq_correlation": corr,
                "beta_noise_ignored": bool(beta_noise),
                "final_sweep_action": action,
                "evidence": (
                    f"top-flow rank {row.get('rank')}; net premium {row.get('net_premium')}; "
                    f"benchmark {corr_row.get('benchmark', '') if not corr_row.empty else ''}; {corr_row.get('reason', '') if not corr_row.empty else ''}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_portfolio_repair(portfolio: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for action in (portfolio or {}).get("risk_actions", []):
        verb = _clean(action.get("action")).upper() or "REVIEW"
        if verb not in {"HOLD", "CLOSE", "ROLL", "TAKE PROFIT", "REDUCE", "SET STOP"}:
            continue
        rows.append(
            {
                "ticker": _clean(action.get("ticker")),
                "action": verb,
                "position": _clean(action.get("position")),
                "reason": _clean(action.get("reason")),
                "instruction": _clean(action.get("instruction")),
                "portfolio impact": _clean(action.get("portfolio_impact")),
                "assignment risk": _clean(action.get("assignment_risk")),
            }
        )
    for action in (portfolio or {}).get("portfolio_income_actions", []):
        rows.append(
            {
                "ticker": _clean(action.get("ticker")),
                "action": _clean(action.get("action")).upper(),
                "position": _clean(action.get("position")),
                "reason": _clean(action.get("reason")),
                "instruction": _clean(action.get("instruction")),
                "portfolio impact": _clean(action.get("portfolio_impact")),
                "assignment risk": _clean(action.get("assignment_risk")),
            }
        )
    return pd.DataFrame(rows)


def build_v4_target_model(
    *,
    asof: dt.date,
    tickets: pd.DataFrame,
    portfolio: dict[str, Any],
    monthly_profit_target: float,
    month_to_date_realized_pnl: float,
    open_unrealized_pnl: float,
    risk_budget: float,
) -> dict[str, Any]:
    days = business_days_remaining(asof)
    weeks = max(1.0, days / 5.0)
    target = safe_float(monthly_profit_target, 10_000.0)
    realized = safe_float(month_to_date_realized_pnl, 0.0)
    unrealized = safe_float(open_unrealized_pnl, 0.0)
    remaining = max(0.0, target - realized - unrealized)
    required_daily = remaining / days
    required_weekly = remaining / weeks
    risk_budget_value = safe_float(risk_budget, 0.0)
    aggregate_cap_applied = math.isfinite(risk_budget_value) and risk_budget_value > 0
    if tickets is None or tickets.empty:
        execute_profit = swing_profit = risk_all = fill_adjusted = 0.0
        risk_bounded_count = 0
        risk_bounded_risk = 0.0
        risk_bounded_fill_adjusted = 0.0
        avg_profit = math.nan
        required_ticket_count = None
        family_stats: dict[str, Any] = {}
    else:
        profit = tickets["profit target"].map(_money_value).fillna(0.0)
        risk = tickets["max loss"].map(_money_value).fillna(0.0)
        disp = tickets["final disposition"].astype(str)
        execute_profit = float(profit[disp.eq("Execute")].sum())
        swing_profit = float(profit.sum())
        risk_all = float(risk.sum())
        multipliers = disp.map(
            {
                "Execute": 1.00,
                "Swing Target / Work Limit": 0.55,
                "Scout": 0.35,
                "Wheel/Cash": 0.30,
                "Portfolio Repair": 0.0,
                "Research": 0.0,
                "Avoid": 0.0,
            }
        ).fillna(0.25)
        fill_adjusted = float((profit * multipliers).sum())
        if aggregate_cap_applied:
            risk_available_prelim = risk_budget_value
            cash_prelim = safe_float((portfolio or {}).get("cash"))
            if math.isfinite(cash_prelim) and cash_prelim > 0:
                risk_available_prelim = min(risk_available_prelim, cash_prelim)
            risk_bounded_count = 0
            risk_bounded_risk = 0.0
            risk_bounded_fill_adjusted = 0.0
            for row_idx in tickets.index:
                row_risk = safe_float(risk.loc[row_idx], 0.0)
                row_fill = safe_float((profit * multipliers).loc[row_idx], 0.0)
                if row_risk <= 0:
                    continue
                if risk_bounded_risk + row_risk <= risk_available_prelim:
                    risk_bounded_risk += row_risk
                    risk_bounded_fill_adjusted += row_fill
                    risk_bounded_count += 1
        else:
            risk_bounded_count = int(len(tickets))
            risk_bounded_risk = risk_all
            risk_bounded_fill_adjusted = fill_adjusted
        avg_profit = float(profit[profit > 0].mean()) if (profit > 0).any() else math.nan
        required_ticket_count = math.ceil(required_daily / (fill_adjusted / max(1, len(tickets)))) if fill_adjusted > 0 else None
        family_stats = {}
        for family, part in tickets.groupby("setup family"):
            wins = part["expected win rate"].astype(str).str.rstrip("%").map(safe_float)
            wins = wins / 100.0
            family_stats[str(family)] = {
                "ticket_count": int(len(part)),
                "expected_win_rate": round(float(wins.mean()), 4) if wins.notna().any() else None,
                "average_profit_target": round(float(part["profit target"].map(_money_value).fillna(0.0).mean()), 2),
                "average_max_loss": round(float(part["max loss"].map(_money_value).fillna(0.0).mean()), 2),
            }
    if aggregate_cap_applied:
        risk_available = risk_budget_value
        cash = safe_float((portfolio or {}).get("cash"))
        if math.isfinite(cash) and cash > 0:
            risk_available = min(risk_available, cash)
    else:
        risk_available = risk_all
    expected_run_rate = risk_bounded_fill_adjusted * days
    required_average_profit = remaining / max(1, len(tickets)) if tickets is not None and not tickets.empty else required_daily
    if remaining <= 0:
        feasibility = "feasible"
        blocker = "monthly target already met"
    elif tickets is None or tickets.empty:
        feasibility = "infeasible"
        blocker = "no swing target tickets survived V4 no-miss construction"
    elif risk_bounded_fill_adjusted >= required_daily and expected_run_rate >= remaining:
        feasibility = "stretched" if execute_profit <= 0 or (aggregate_cap_applied and risk_all > risk_available) else "feasible"
        if aggregate_cap_applied and risk_all > risk_available:
            execute_clause = (
                f"current Execute profit potential ${execute_profit:,.2f}"
                if execute_profit > 0
                else "zero current Execute profit potential"
            )
            blocker = (
                f"risk budget bottleneck: all visible tickets require ${risk_all:,.2f} max loss, "
                f"but allowed risk budget is ${risk_available:,.2f}; in-budget fill-adjusted potential is "
                f"${risk_bounded_fill_adjusted:,.2f}; {execute_clause}"
            )
        elif execute_profit <= 0:
            blocker = "zero Execute; target math depends on next-session fills at the listed limits"
        else:
            blocker = "V4 found enough visible target potential; execution depends on fills and per-ticket risk discipline"
    elif expected_run_rate >= remaining * 0.60:
        feasibility = "stretched"
        label = "risk-bounded" if aggregate_cap_applied else "visible"
        blocker = f"{label} fill-adjusted potential ${risk_bounded_fill_adjusted:,.2f} is below required daily ${required_daily:,.2f}"
    else:
        feasibility = "infeasible"
        label = "risk-bounded" if aggregate_cap_applied else "visible"
        blocker = f"{label} fill-adjusted potential ${risk_bounded_fill_adjusted:,.2f} is below required daily ${required_daily:,.2f}"
    sizing_can_close = feasibility in {"feasible", "stretched"} and (not aggregate_cap_applied or risk_all <= risk_available)
    if aggregate_cap_applied and risk_all > risk_available and risk_bounded_fill_adjusted >= required_daily:
        sizing_can_close = True
    if execute_profit < required_daily:
        sizing_can_close = False
        blocker = (
            "approved Execute profit is below the required daily pace; "
            "theoretical Work Limit inventory is not a forecast"
        )
    return {
        "monthly_profit_target": round(target, 2),
        "month_to_date_realized_pnl": round(realized, 2),
        "open_unrealized_pnl": round(unrealized, 2),
        "remaining_monthly_target": round(remaining, 2),
        "business_days_remaining": days,
        "required_daily_pl": round(required_daily, 2),
        "required_weekly_pl": round(required_weekly, 2),
        "execute_profit_potential": round(execute_profit, 2),
        "swing_target_profit_potential_if_filled": round(swing_profit, 2),
        "theoretical_target_inventory_fill_adjusted_potential": round(fill_adjusted, 2),
        "realistic_fill_adjusted_target_potential": round(execute_profit, 2),
        "risk_bounded_target_ticket_count": int(risk_bounded_count),
        "risk_bounded_max_loss": round(risk_bounded_risk, 2),
        "risk_bounded_fill_adjusted_target_potential": round(risk_bounded_fill_adjusted, 2),
        "max_risk_if_all_target_tickets_fill": round(risk_all, 2),
        "risk_available": round(risk_available, 2),
        "aggregate_risk_budget_applied": bool(aggregate_cap_applied),
        "aggregate_risk_budget_input": round(risk_budget_value, 2) if math.isfinite(risk_budget_value) else 0.0,
        "risk_policy_summary": (
            "Optional aggregate risk budget applied to target math."
            if aggregate_cap_applied
            else "No aggregate slate risk budget configured; V4 shows every valid Execute/Target. The per-ticket 2% account-loss level is advisory and remains visible in the risk audit."
        ),
        "required_average_profit_per_trade": round(required_average_profit, 2),
        "required_number_of_target_tickets": int(required_ticket_count) if required_ticket_count is not None else None,
        "expected_by_setup_family": family_stats,
        "theoretical_target_inventory_monthly_run_rate": round(expected_run_rate, 2),
        "expected_monthly_run_rate": round(execute_profit * days, 2),
        "sizing_can_close_gap": bool(sizing_can_close),
        "target_feasibility": feasibility if execute_profit >= required_daily else "not demonstrated",
        "exact_blocker_to_10k_month": blocker,
        "average_target_profit_per_ticket": round(avg_profit, 2) if math.isfinite(avg_profit) else None,
    }


def _market_insight_lines(regime_context: dict[str, Any], target_model: dict[str, Any], tickets: pd.DataFrame) -> list[str]:
    base = regime_context.get("base_regime") or regime_context or {}
    indices = regime_context.get("indices") or {}
    sectors = regime_context.get("sector_etfs") or {}
    rates = regime_context.get("rates_yields") or {"status": "unavailable"}
    mag7 = regime_context.get("mag7_leadership") or {}
    semi = regime_context.get("semi_leadership") or {}
    sector_rows = sorted(
        [(ticker, safe_float(data.get("return_1d")), safe_float(data.get("flow_bias"))) for ticker, data in sectors.items()],
        key=lambda item: item[1] if math.isfinite(item[1]) else -999,
        reverse=True,
    )
    leaders = ", ".join(f"{ticker} {ret:.1%}" for ticker, ret, _ in sector_rows[:3] if math.isfinite(ret)) or "unavailable"
    laggards = ", ".join(f"{ticker} {ret:.1%}" for ticker, ret, _ in sector_rows[-3:] if math.isfinite(ret)) or "unavailable"
    index_text = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        item = indices.get(ticker) or {}
        ret = safe_float(item.get("return_1d"))
        flow = safe_float(item.get("flow_bias"))
        index_text.append(f"{ticker} {ret:.1%} flow {flow:.2f}" if math.isfinite(ret) else f"{ticker} unavailable")
    lane_counts = tickets["lane"].value_counts().to_dict() if tickets is not None and not tickets.empty else {}
    best_edge = "directional debit momentum" if lane_counts.get("Momentum Debit", 0) > lane_counts.get("Swing Target / Work Limit", 0) else "defined-risk swing targets and selective premium selling"
    if target_model.get("execute_profit_potential", 0.0) <= 0:
        best_edge = "target-price swing work, not immediate Execute"
    regime_sync = (
        "aligned"
        if base.get("trend") in {"uptrend", "range"} and lane_counts
        else "conflicted"
        if base.get("trend") and base.get("flow") and base.get("trend") != base.get("flow")
        else "unknown"
    )
    return [
        f"- Regime-Synchronization: {regime_sync}; trend={base.get('trend', 'unknown')}, flow={base.get('flow', 'unknown')}, volatility={base.get('volatility', 'unknown')}.",
        f"- SPY / QQQ / IWM trend: {'; '.join(index_text)}.",
        f"- VIX / volatility regime: {base.get('vix_proxy', 'n/a')} proxy, {base.get('volatility', 'unknown')} volatility, {base.get('trend', 'unknown')} tape.",
        f"- Sector leadership and laggards: leaders {leaders}; laggards {laggards}.",
        f"- Mag7 / semi / AI concentration: Mag7 status {mag7.get('status', 'unknown')} avg return {safe_float(mag7.get('avg_return_1d')):.1%}; semi status {semi.get('status', 'unknown')} avg return {safe_float(semi.get('avg_return_1d')):.1%}.",
        f"- Rates/yields: {rates.get('status', 'unavailable')} - {rates.get('reason', 'no local rates/yields feed available')}.",
        "- Bullish playbook: favor defined-risk bullish spreads only at target limits, with semis/large-cap tech confirmed by fresh quote width and OI carryover.",
        "- Bearish playbook: use bear-call or bear-put structures only where top-flow/relative weakness supports it; do not fight broad tape strength without a hard catalyst.",
        "- Rangebound/income playbook: prioritize credit spreads and Wheel/Cash only where assignment quality, liquidity, and portfolio concentration pass.",
        f"- Setups favored tomorrow: {', '.join(sorted(lane_counts)) if lane_counts else 'none'}.",
        "- Setups to avoid tomorrow: near-earnings structures, extreme bid/ask width, negative-edge families, and any ticket requiring a chase beyond the listed target.",
        f"- Market regime conflicts: price trend is {base.get('trend', 'unknown')} while flow is {base.get('flow', 'unknown')}; macro/news gates must clear before size-up.",
        f"- Best edge classification: {best_edge}.",
    ]


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None, *, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"
    shown = df.copy()
    if columns is not None:
        shown = shown[[col for col in columns if col in shown.columns]]
    if max_rows is not None:
        shown = shown.head(max_rows)
    shown = shown.where(pd.notna(shown), "")
    return shown.to_markdown(index=False)


def _compact_ticket_table(tickets: pd.DataFrame) -> pd.DataFrame:
    columns = ["Decision", "Trade", "Entry", "Confidence", "Target / Risk", "Action"]
    if tickets is None or tickets.empty:
        return pd.DataFrame(columns=columns)

    def action_text(row: pd.Series) -> str:
        disposition = _clean(row.get("final disposition"))
        blocker = _clean(row.get("blocker before entry"))
        if disposition == "Execute":
            return "Requote first; enter only if the limit still holds; attach OCO"
        if disposition == "Scout":
            return (blocker[:157] + "...") if len(blocker) > 160 else (blocker or "Review only; no order authority")
        return (blocker[:157] + "...") if len(blocker) > 160 else (blocker or "Work the stated limit; do not chase")

    rows: list[dict[str, str]] = []
    for _, row in tickets.iterrows():
        ticker = _clean(row.get("ticker")).upper()
        expiry = _clean(row.get("expiry"))
        dte = _clean(row.get("DTE"))
        legs = _clean(row.get("trade legs"))
        if ticker and expiry:
            legs = legs.replace(f"{ticker} {expiry} ", "")
        legs = legs.replace("sell ", "SELL ").replace("buy ", "BUY ")
        trade = f"**{ticker}** - {legs}"
        if expiry:
            trade += f" - Exp {expiry}"
        if dte:
            trade += f" ({dte} DTE)"
        entry_target = _clean(row.get("next-session swing entry target"))
        live_reference = _clean(row.get("current Schwab mid/natural reference"))
        entry = f"Limit {entry_target}; Schwab mid/natural {live_reference}"
        risk = f"Take {_clean(row.get('profit target'))}; max loss {_clean(row.get('max loss'))}"
        rows.append(
            {
                "Decision": _clean(row.get("display status")),
                "Trade": trade,
                "Entry": entry,
                "Confidence": _clean(row.get("confidence rating")),
                "Target / Risk": risk,
                "Action": action_text(row),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _readable_action_ticket_table(tickets: pd.DataFrame) -> pd.DataFrame:
    """Render a complete but scan-friendly action table without a 30+ column wall."""
    columns = [
        "Ticket",
        "Strategy and explicit legs",
        "Expiry / DTE",
        "Entry target / Schwab reference",
        "Confidence / risk / action",
    ]
    if tickets is None or tickets.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, str]] = []
    for _, row in tickets.iterrows():
        expected_value = _clean(row.get("expected value"))
        profit_factor = _clean(row.get("implied profit factor"))
        blocker = _clean(row.get("blocker before entry"))
        rows.append(
            {
                "Ticket": (
                    f"#{_clean(row.get('rank'))} **{_clean(row.get('ticker')).upper()}**<br>"
                    f"{_clean(row.get('display status'))}"
                ),
                "Strategy and explicit legs": _clean(row.get("trade legs")),
                "Expiry / DTE": f"{_clean(row.get('expiry'))}<br>{_clean(row.get('DTE'))} DTE",
                "Entry target / Schwab reference": (
                    f"**Limit:** {_clean(row.get('next-session swing entry target'))}<br>"
                    f"**Mid / natural:** {_clean(row.get('current Schwab mid/natural reference'))}"
                ),
                "Confidence / risk / action": (
                    f"{_clean(row.get('confidence rating'))}<br>"
                    f"Win {_clean(row.get('expected win rate'))}; {_clean(row.get('confidence evidence'))}<br>"
                    f"{_clean(row.get('suggested size'))}<br>"
                    f"Max profit {_clean(row.get('maximum profit'))}; take {_clean(row.get('profit target'))}; "
                    f"1-contract risk {_clean(row.get('one-contract risk') or row.get('max loss'))}; "
                    f"R/R {_clean(row.get('reward/risk'))}; EV {expected_value}; PF {profit_factor}<br>"
                    f"{_clean(row.get('flow/OI evidence'))}<br>"
                    f"{_clean(row.get('GEX context'))}<br>"
                    f"{_clean(row.get('liquidity context'))}<br>"
                    f"{_clean(row.get('portfolio context'))}<br>"
                    f"{blocker or _clean(row.get('manual review instruction'))}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _readable_ticket_detail_lines(tickets: pd.DataFrame) -> list[str]:
    """Return vertical ticket cards for actionable and target-only structures."""
    if tickets is None or tickets.empty:
        return []
    lines: list[str] = []
    for _, row in tickets.iterrows():
        rank = _clean(row.get("rank"))
        ticker = _clean(row.get("ticker")).upper()
        confidence = _clean(row.get("confidence rating"))
        safety_flags = str(row.get("safety calibration flags") or "").strip()
        if not safety_flags or safety_flags.lower() in {"nan", "nat"}:
            safety_flags = "None"
        details = pd.DataFrame(
            [
                {"Field": "Status / confidence", "Value": f"{_clean(row.get('display status'))} — {confidence}"},
                {"Field": "Strategy and explicit legs", "Value": _clean(row.get("trade legs"))},
                {"Field": "Expiry / DTE", "Value": f"{_clean(row.get('expiry'))} / {_clean(row.get('DTE'))} DTE"},
                {"Field": "Entry target", "Value": _clean(row.get("next-session swing entry target"))},
                {"Field": "Schwab mid / executable natural", "Value": _clean(row.get("current Schwab mid/natural reference"))},
                {"Field": "Suggested size", "Value": _clean(row.get("suggested size"))},
                {
                    "Field": "Maximum profit / take-profit / one-contract risk / reward-risk",
                    "Value": (
                        f"{_clean(row.get('maximum profit'))} / {_clean(row.get('profit target'))} / "
                        f"{_clean(row.get('one-contract risk') or row.get('max loss'))} / "
                        f"{_clean(row.get('reward/risk'))}"
                    ),
                },
                {
                    "Field": "Expected win rate / basis",
                    "Value": f"{_clean(row.get('expected win rate'))} — {_clean(row.get('win-rate basis'))}",
                },
                {"Field": "Confidence evidence", "Value": _clean(row.get("confidence evidence"))},
                {"Field": "Payoff evidence", "Value": _clean(row.get("payoff evidence"))},
                {
                    "Field": "Expected avg win / avg loss / EV / PF",
                    "Value": (
                        f"{_clean(row.get('expected average win'))} / {_clean(row.get('expected average loss'))} / "
                        f"{_clean(row.get('expected value'))} / {_clean(row.get('implied profit factor'))}"
                    ),
                },
                {"Field": "Flow / exact-leg OI", "Value": _clean(row.get("flow/OI evidence"))},
                {"Field": "Schwab chain GEX", "Value": _clean(row.get("GEX context"))},
                {"Field": "Liquidity", "Value": _clean(row.get("liquidity context"))},
                {"Field": "Portfolio overlap", "Value": _clean(row.get("portfolio context"))},
                {
                    "Field": "Trend / regime",
                    "Value": f"{_clean(row.get('EOD trend evidence'))}; {_clean(row.get('regime fit'))}",
                },
                {"Field": "Catalyst / earnings", "Value": _clean(row.get("catalyst/earnings risk"))},
                {"Field": "OCO order logic", "Value": _clean(row.get("OCO bracket order logic"))},
                {"Field": "Stop / invalidation", "Value": _clean(row.get("stop/invalidation"))},
                {"Field": "Gap-risk plan", "Value": _clean(row.get("gap-risk plan +/-1% open"))},
                {"Field": "Blocker before entry", "Value": _clean(row.get("blocker before entry"))},
                {"Field": "Manual pre-order review", "Value": _clean(row.get("manual review instruction"))},
                {"Field": "Safety flags", "Value": safety_flags},
            ]
        )
        lines.extend(
            [
                "",
                f"### #{rank} {ticker} — {confidence}",
                "",
                _markdown_table(details),
            ]
        )
    return lines


def _attach_schwab_gex_context(tickets: pd.DataFrame, chain_dir: Path) -> pd.DataFrame:
    """Attach read-only Schwab chain gamma-times-OI context to ticket rows."""
    out = tickets.copy()
    if out.empty:
        out["GEX context"] = pd.Series(dtype=str)
        return out
    try:
        from uwos.run_mode_a_two_stage import compute_schwab_chain_gex
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        out["GEX context"] = f"UNAVAILABLE - Schwab GEX calculator import failed: {exc}"
        return out

    contexts: dict[str, str] = {}
    for ticker in out.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper().unique():
        if not ticker:
            continue
        path = chain_dir / f"{ticker}.json"
        if not path.exists():
            contexts[ticker] = (
                "UNAVAILABLE - same-session Schwab chain GEX snapshot missing; "
                "recheck before entry"
            )
            continue
        try:
            result = compute_schwab_chain_gex(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            contexts[ticker] = f"UNAVAILABLE - Schwab chain GEX calculation failed: {exc}"
            continue
        if not result:
            contexts[ticker] = (
                "UNAVAILABLE - Schwab chain lacked usable gamma/open-interest rows; "
                "recheck before entry"
            )
            continue
        net_gex = safe_float(result.get("net_gex"))
        support = safe_float(result.get("gex_support"))
        resistance = safe_float(result.get("gex_resistance"))
        net_text = (
            f"{'+' if net_gex >= 0 else '-'}${abs(net_gex) / 1_000_000:.1f}M"
            if math.isfinite(net_gex)
            else "unavailable"
        )
        support_text = f"{support:g}" if math.isfinite(support) else "unavailable"
        resistance_text = f"{resistance:g}" if math.isfinite(resistance) else "unavailable"
        snapshot = _clean(result.get("gex_time")) or "time unavailable"
        contexts[ticker] = (
            f"Schwab live-chain gamma×OI proxy: {_clean(result.get('gex_regime')) or 'unknown'}; "
            f"net {net_text}; support {support_text}; resistance {resistance_text}; "
            f"snapshot {snapshot}"
        )
    out["GEX context"] = out.get("ticker", pd.Series("", index=out.index)).fillna("").astype(str).str.upper().map(contexts).fillna(
        "UNAVAILABLE - ticker-level Schwab chain GEX context missing; recheck before entry"
    )
    return out


def _compact_decision_candidate_table(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "status",
        "ticker",
        "trade",
        "dte",
        "decision tier",
        "confirmation",
        "payoff status",
        "per-ticket replay edge",
        "why not promoted",
    ]
    if scored is None or scored.empty or "decision_eligible" not in scored.columns:
        return pd.DataFrame(columns=columns)
    eligible = scored["decision_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    )
    rows = []
    for _, row in scored.loc[eligible].iterrows():
        disposition = _disposition(row, targetable=_targetable(row))
        legacy_payoff_status = _clean(row.get("payoff_calibration_status")) or "unavailable"
        if _directional_credit_execution_ready(row):
            family_status = _clean(
                row.get("directional_credit_family_validation_status")
            ).upper()
            payoff_status = f"PASS (execution policy; family {family_status})"
            legacy_warning = (
                f" Legacy sparse-route status {legacy_payoff_status} is retained for audit but "
                "is non-authoritative for this exact execution-policy population."
                if legacy_payoff_status in {"VETO", "FAIL", "INSUFFICIENT"}
                else ""
            )
            blocker = "No downstream payoff blocker. " + _payoff_evidence_text(row) + legacy_warning
        else:
            payoff_status = legacy_payoff_status
            blocker = (
                _clean(row.get("payoff_calibration_reason"))
                if payoff_status in {"VETO", "FAIL", "INSUFFICIENT"}
                else _clean(row.get("v4_direct_disposition_reason"))
                or _clean(row.get("trade_status_reason"))
                or _clean(row.get("decision_reason"))
            )
        rows.append(
            {
                "status": _status_label(disposition),
                "ticker": _clean(row.get("ticker")).upper(),
                "trade": _trade_legs(row),
                "dte": _clean(row.get("dte")),
                "decision tier": _clean(row.get("decision_tier")) or "unclassified",
                "confirmation": _clean(row.get("v4_confirmation_status")) or "unavailable",
                "payoff status": payoff_status,
                "per-ticket replay edge": _edge_evidence_text(row),
                "why not promoted": blocker[:180] if blocker else "No downstream blocker recorded",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _compact_portfolio_repair_table(portfolio_repair: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "action", "reason", "instruction"]
    if portfolio_repair is None or portfolio_repair.empty:
        return pd.DataFrame(columns=columns)
    out = portfolio_repair[[col for col in columns if col in portfolio_repair.columns]].copy()
    for col, length in [("reason", 90), ("instruction", 110)]:
        if col in out.columns:
            out[col] = out[col].map(lambda value: _clean(value)[:length])
    return out


def _compact_no_miss_table(no_miss: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "top_flow_rank",
        "ticker",
        "final_disposition",
        "shown_as_target_ticket",
        "if_not_targetable_exact_reason",
    ]
    if no_miss is None or no_miss.empty:
        return pd.DataFrame(columns=columns)
    out = no_miss[[col for col in columns if col in no_miss.columns]].copy()
    if "if_not_targetable_exact_reason" in out.columns:
        out["if_not_targetable_exact_reason"] = out["if_not_targetable_exact_reason"].map(lambda value: _clean(value)[:120])
    return out


def _compact_opportunity_board(board: pd.DataFrame) -> pd.DataFrame:
    columns = ["Status", "Ticker", "Trade", "Entry limit", "Max loss", "Required confirmation"]
    if board is None or board.empty:
        return pd.DataFrame(columns=columns)
    out = board[[col for col in columns if col in board.columns]].copy()
    for col, length in [("Trade", 90), ("Required confirmation", 120)]:
        if col in out.columns:
            out[col] = out[col].map(lambda value: _clean(value)[:length])
    return out


def _target_status_panel(target_model: dict[str, Any], counts: dict[str, int]) -> pd.DataFrame:
    risk_all = safe_float(target_model.get("max_risk_if_all_target_tickets_fill"), 0.0)
    risk_available = safe_float(target_model.get("risk_available"), 0.0)
    risk_bounded = safe_float(target_model.get("risk_bounded_fill_adjusted_target_potential"), 0.0)
    execute_profit = safe_float(target_model.get("execute_profit_potential"), 0.0)
    aggregate_cap = bool(target_model.get("aggregate_risk_budget_applied"))
    if aggregate_cap and risk_all > risk_available > 0:
        bottleneck = "Risk budget"
        plain = f"All visible tickets would risk ${risk_all:,.2f}; allowed budget is ${risk_available:,.2f}."
    elif execute_profit <= 0:
        bottleneck = "No current Execute"
        plain = "Ideas exist, but fills still depend on next-session target prices."
    elif not aggregate_cap:
        bottleneck = "No artificial slate cap"
        plain = "V4 is showing every valid Execute/Target; only per-ticket safety caps apply."
    else:
        bottleneck = "None obvious"
        plain = "Current run-rate can work inside the configured risk budget."
    return pd.DataFrame(
        [
            {
                "Question": "Can V4 find trades?",
                "Answer": (
                    f"Yes: {counts.get('execute', 0)} Execute, {counts.get('scout', 0)} Scout, "
                    f"{counts.get('swing_target_work_limit', 0)} Work Limit."
                    if sum(counts.get(key, 0) for key in ["execute", "scout", "swing_target_work_limit"]) > 0
                    else "No: zero Execute, Scout, or Work Limit rows passed the current evidence and live-entry gates."
                ),
            },
            {"Question": "What blocks $10k/month?", "Answer": bottleneck},
            {"Question": "Plain English", "Answer": plain},
            {"Question": "Visible expected potential", "Answer": _money(risk_bounded)},
            {"Question": "Immediate Execute profit potential", "Answer": _money(execute_profit)},
        ]
    )


def _artifact_path(out_dir: Path, stem: str, asof: dt.date, suffix: str = "csv") -> Path:
    return out_dir / f"codexdaily_v4_{stem}_{asof}.{suffix}"


def _schwab_status(data_quality: dict[str, Any]) -> str:
    items = data_quality.get("items") or []
    quote = next((item for item in items if item.get("check") == "Schwab quotes available"), {})
    portfolio = next((item for item in items if item.get("check") == "Schwab portfolio available"), {})
    return f"quotes={quote.get('status', 'unknown')} ({quote.get('detail', '')}); portfolio={portfolio.get('status', 'unknown')} ({portfolio.get('detail', '')})"


def _public_opportunity_board(board: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "Modeled win rate",
        "Win-rate basis",
        "Confidence evidence",
        "Payoff evidence",
        "Per-ticket replay edge",
        "Post-pricing EV / PF",
        "Expected value source",
        "Edge sample size / win rate / avg P/L",
        "Required confirmation",
        "Why Execute, Scout, Research, or Avoid",
    ]
    if board is None or board.empty:
        return pd.DataFrame(columns=columns)
    public = board[[col for col in columns if col in board.columns]].copy()
    for col in columns:
        if col not in public.columns:
            public[col] = ""
    return public[columns]


def _risk_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_risk_per_trade": args.max_risk_per_trade,
        "max_risk_per_day": args.max_risk_per_day,
        "max_open_risk_by_ticker": args.max_open_risk_by_ticker,
        "max_correlated_sector_exposure": args.max_correlated_sector_exposure,
        "max_total_open_risk": args.max_total_open_risk,
        "max_contracts_per_trade": args.max_contracts_per_trade,
        "minimum_expected_value_per_dollar_risk": args.minimum_expected_value_per_dollar_risk,
        "monthly_profit_target": args.monthly_profit_target,
        "daily_loss_limit": args.daily_loss_limit,
        "risk_mandate": args.risk_mandate,
        "index_income_mode": args.index_income_mode,
        "portfolio_income_mode": args.portfolio_income_mode,
        "covered_income_allowed_tickers": [
            ticker.strip().upper()
            for ticker in str(args.covered_income_allowed_tickers or "").split(",")
            if ticker.strip()
        ],
        "allow_new_trades": True,
    }


def _return_1d(row: pd.Series) -> float:
    close = safe_float(row.get("close"))
    prev = safe_float(row.get("prev_close"))
    if not math.isfinite(close) or not math.isfinite(prev) or prev == 0:
        return math.nan
    return close / prev - 1.0


def _build_v4_regime_context(
    *,
    stock_screener: pd.DataFrame,
    base_regime: dict[str, Any],
    liquidity_shift: dict[str, Any],
    asof: dt.date,
    run_mode: str,
) -> dict[str, Any]:
    df = stock_screener.copy() if stock_screener is not None else pd.DataFrame()
    if not df.empty and "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    rows = {str(row.get("ticker")): row for _, row in df.iterrows()} if not df.empty else {}

    def index_item(ticker: str) -> dict[str, Any]:
        row = rows.get(ticker)
        if row is None:
            return {"status": "unavailable"}
        ret = _return_1d(row)
        return {
            "close": safe_float(row.get("close")),
            "prev_close": safe_float(row.get("prev_close")),
            "return_1d": round(ret, 4) if math.isfinite(ret) else None,
            "flow_bias": safe_float(row.get("flow_bias")),
            "iv30d": safe_float(row.get("iv30d")),
        }

    sector_etfs = {}
    for ticker in ["SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC"]:
        row = rows.get(ticker)
        if row is None:
            continue
        ret = _return_1d(row)
        sector_etfs[ticker] = {
            "return_1d": round(ret, 4) if math.isfinite(ret) else None,
            "flow_bias": safe_float(row.get("flow_bias")),
        }

    def leadership(tickers: list[str]) -> dict[str, Any]:
        values = []
        for ticker in tickers:
            row = rows.get(ticker)
            if row is None:
                continue
            ret = _return_1d(row)
            if math.isfinite(ret):
                values.append(ret)
        if not values:
            return {"status": "unavailable", "avg_return_1d": None}
        return {"status": "ok", "avg_return_1d": round(sum(values) / len(values), 4)}

    top_flow = liquidity_shift.get("top_flow_universe") if isinstance(liquidity_shift, dict) else pd.DataFrame()
    zero_dte = liquidity_shift.get("zero_dte_gamma") if isinstance(liquidity_shift, dict) else pd.DataFrame()
    return {
        "pipeline": PIPELINE_NAME_V4,
        "asof": str(asof),
        "run_mode": run_mode,
        "base_regime": base_regime,
        "indices": {ticker: index_item(ticker) for ticker in ["SPY", "QQQ", "IWM"]},
        "vix": {"proxy": base_regime.get("vix_proxy"), "volatility_regime": base_regime.get("volatility")},
        "sector_etfs": sector_etfs,
        "rates_yields": {"status": "unavailable", "reason": "no local rates/yields feed found in UW exports"},
        "mag7_leadership": leadership(["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]),
        "semi_leadership": leadership(["NVDA", "AMD", "AVGO", "TSM", "MU", "SMCI", "SMH"]),
        "liquidity_shift_thresholds": (liquidity_shift or {}).get("thresholds", {}),
        "top_flow_tickers": top_flow["ticker"].head(10).tolist() if isinstance(top_flow, pd.DataFrame) and not top_flow.empty else [],
        "zero_dte_gamma_context": zero_dte.head(10).to_dict(orient="records") if isinstance(zero_dte, pd.DataFrame) and not zero_dte.empty else [],
    }


def _v4_current_entry_price(row: pd.Series | dict[str, Any]) -> float:
    if _is_debit(row):
        return safe_float(row.get("natural_debit"), safe_float(row.get("mid_debit"), safe_float(row.get("debit"))))
    return safe_float(row.get("natural_credit"), safe_float(row.get("mid_credit"), safe_float(row.get("credit"))))


def _combination_quote_width_ratio(row: pd.Series | dict[str, Any]) -> float:
    if _is_debit(row):
        mid = safe_float(row.get("mid_debit"))
        natural = safe_float(row.get("natural_debit"))
    else:
        mid = safe_float(row.get("mid_credit"))
        natural = safe_float(row.get("natural_credit"))
    width = safe_float(row.get("spread_width"))
    if math.isfinite(mid) and math.isfinite(natural) and math.isfinite(width) and width > 0:
        return abs(mid - natural) / width
    return math.nan


def _v4_entry_target_value(row: pd.Series | dict[str, Any]) -> float:
    target = safe_float(row.get("required_entry"), safe_float(row.get("target_entry")))
    if math.isfinite(target):
        return target
    return _entry_price_from_target(row)


def _v4_target_met(row: pd.Series | dict[str, Any]) -> bool:
    current = _v4_current_entry_price(row)
    target = _v4_entry_target_value(row)
    if not math.isfinite(current) or not math.isfinite(target) or target <= 0:
        return False
    safe_target = _expectancy_safe_entry_price(row)
    if math.isfinite(safe_target):
        target = safe_target
    return current <= target if _is_debit(row) else current >= target


def _v4_nonnegative_ev(row: pd.Series | dict[str, Any]) -> bool:
    if _medium_debit_sleeve_eligible(row):
        expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
        return bool(
            math.isfinite(expected_value)
            and expected_value > 0
            and not math.isnan(profit_factor)
            and profit_factor >= V4_EXECUTE_MIN_PROFIT_FACTOR
        )
    if _payoff_evidence_ready(row):
        expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
        return bool(
            math.isfinite(expected_value)
            and expected_value > 0
            and not math.isnan(profit_factor)
            and profit_factor >= V4_EXECUTE_MIN_PROFIT_FACTOR
        )
    replay = _clean(row.get("replay_ev_verdict")).lower()
    edge = _clean(row.get("edge_verdict")).lower()
    if replay.startswith("negative") or edge == "negative":
        return False
    avg = safe_float(row.get("edge_avg_pnl"))
    if math.isfinite(avg):
        return avg > 0
    if _is_credit(row) and (replay == "acceptable_secondary_income" or edge == "acceptable_secondary_income"):
        sample = safe_float(row.get("edge_sample_size"))
        raw_win = safe_float(row.get("edge_win_rate"))
        win = safe_float(row.get("edge_effective_win_rate"), raw_win)
        if math.isfinite(sample) and sample < 30:
            return False
        if math.isfinite(win) and win < 0.58:
            return False
    return edge in {"positive", "acceptable", "acceptable_secondary_income", "thin_sample", "thin"}


def apply_v4_professional_dispositions(scored: pd.DataFrame, *, asof: dt.date | None = None) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = _normalize_v4_dataframe(scored)
    if asof is not None:
        out["v4_asof"] = asof.isoformat()
    for col in ["trade_status", "trade_tier", "trade_status_reason", "v4_direct_disposition_reason"]:
        if col not in out.columns:
            out[col] = ""
    if "v4_pre_disposition_status" not in out.columns:
        out["v4_pre_disposition_status"] = out["trade_status"]
    if "debit_policy_tier" not in out.columns:
        out["debit_policy_tier"] = ""
    if "credit_policy_tier" not in out.columns:
        out["credit_policy_tier"] = ""
    if "contracts" not in out.columns:
        out["contracts"] = 0
    if "v4_execution_authority" not in out.columns:
        out["v4_execution_authority"] = ""
    for col in [
        "v4_post_pricing_expected_value",
        "v4_post_pricing_profit_factor",
        "v4_live_win_payoff",
        "v4_live_loss_payoff",
        "v4_effective_win_rate",
        "v4_expectancy_safe_entry_price",
    ]:
        if col not in out.columns:
            out[col] = math.nan
    out["v4_expectancy_policy_version"] = V4_EXPECTANCY_POLICY_VERSION
    for idx, row in out.iterrows():
        if _symbol_credit_execution_ready(row):
            out.at[idx, "contracts"] = 1
            out.at[idx, "v4_execution_authority"] = "symbol_regime_credit_one_lot"
        elif _walk_forward_credit_execution_ready(row):
            out.at[idx, "contracts"] = 1
            out.at[idx, "v4_execution_authority"] = "walk_forward_credit_one_lot"
        elif _medium_debit_sleeve_eligible(row):
            out.at[idx, "contracts"] = 1
            out.at[idx, "v4_execution_authority"] = "validated_medium_debit_one_lot"
        expected_value, profit_factor, win_payoff, loss_payoff = _post_pricing_expectancy(row)
        out.at[idx, "v4_post_pricing_expected_value"] = expected_value
        out.at[idx, "v4_post_pricing_profit_factor"] = profit_factor
        out.at[idx, "v4_live_win_payoff"] = win_payoff
        out.at[idx, "v4_live_loss_payoff"] = loss_payoff
        out.at[idx, "v4_effective_win_rate"] = _effective_win_rate(row)
        out.at[idx, "v4_expectancy_safe_entry_price"] = _expectancy_safe_entry_price(row)
        hard = _hard_blocker_reason(row)
        if hard:
            out.at[idx, "trade_status"] = "Avoid"
            out.at[idx, "trade_tier"] = "Avoid"
            out.at[idx, "v4_direct_disposition_reason"] = f"V4 hard blocker: {hard}"
            continue
        safety = _safety_research_reason(row)
        if safety:
            out.at[idx, "trade_status"] = "Research"
            out.at[idx, "trade_tier"] = "Research"
            out.at[idx, "v4_direct_disposition_reason"] = safety
            continue
        target_met = _v4_target_met(row)
        quote_width = safe_float(row.get("quote_width_pct"), 9.0)
        combination_width = _combination_quote_width_ratio(row)
        if math.isfinite(combination_width):
            quote_width = max(quote_width, combination_width)
        quote_width_limit = (
            V4_MEDIUM_DEBIT_MAX_QUOTE_WIDTH
            if _medium_debit_sleeve_eligible(row)
            else 0.20
        )
        ev_ok = _v4_nonnegative_ev(row)
        has_risk = math.isfinite(_max_loss_value(row)) and _max_loss_value(row) > 0 and math.isfinite(_target_profit_value(row))
        execution_quote_blocker = _execution_quote_blocker(row)
        execution_catalyst_blocker = _execution_catalyst_verification_blocker(row)
        identification_quality_blocker = _execute_quality_blocker(
            row,
            include_entry_price=False,
            include_execution_quote=False,
            include_catalyst_verification=False,
        )
        preprice_quality_blocker = (
            execution_quote_blocker
            or execution_catalyst_blocker
            or identification_quality_blocker
        )
        entry_price_blocker = _expectancy_safe_price_blocker(row)
        execute_quality_blocker = preprice_quality_blocker or entry_price_blocker
        if _is_debit(row) and not _medium_debit_sleeve_eligible(row):
            identification_quality_blocker = (
                "debit execution disabled: corrected next-session replay PF is below 1 "
                "for both Bull Call and Bear Put families"
            )
            preprice_quality_blocker = identification_quality_blocker
            execute_quality_blocker = preprice_quality_blocker
        debit_tier = ""
        credit_tier = ""
        if _is_debit(row):
            debit_tier, _ = debit_spread_confidence(row, live=True)
            if _medium_debit_sleeve_eligible(row):
                debit_tier = "medium"
            else:
                debit_tier = "disabled"
            if debit_tier == "high" and not confidence_high_ready(row):
                debit_tier = "medium"
            out.at[idx, "debit_policy_tier"] = debit_tier
        elif _is_credit(row):
            credit_row = dict(row)
            if _symbol_credit_execution_ready(row):
                credit_row["regime_trend"] = row.get("symbol_regime_trend")
                credit_row["regime"] = row.get("symbol_regime_trend")
                credit_tier = "probationary"
            elif _walk_forward_credit_execution_ready(row):
                credit_tier = "medium"
            else:
                credit_tier, _ = credit_spread_confidence(credit_row, live=True)
            if credit_tier == "high" and not confidence_high_ready(row):
                credit_tier = "medium"
            if credit_tier == "reject" and _validated_family_evidence(row):
                credit_tier = "validated"
            out.at[idx, "credit_policy_tier"] = credit_tier
        if target_met and ev_ok and has_risk and quote_width <= quote_width_limit and not execute_quality_blocker:
            out.at[idx, "trade_status"] = "Execute"
            out.at[idx, "decision_eligible"] = True
            out.at[idx, "decision_tier"] = "v4-validated-execute"
            probationary = _probationary_execution_ready(row)
            out.at[idx, "trade_tier"] = (
                f"Execute V4 Direct - {debit_tier.title()} Debit" if debit_tier else "Execute V4 Direct"
            )
            if credit_tier:
                out.at[idx, "trade_tier"] = f"Execute V4 Direct - {credit_tier.title()} Credit"
            if probationary:
                symbol_credit = _symbol_credit_execution_ready(row)
                walk_forward_credit = _walk_forward_credit_execution_ready(row)
                out.at[idx, "trade_tier"] = (
                    "Execute V4 Pilot - Symbol Trend Credit - 1 Contract"
                    if symbol_credit
                    else "Execute V4 Medium - Walk-Forward Credit - 1 Contract"
                    if walk_forward_credit
                    else "Execute V4 Pilot - 1 Contract"
                )
                out.at[idx, "contracts"] = 1
                out.at[idx, "v4_execution_authority"] = (
                    "symbol_regime_credit_one_lot"
                    if symbol_credit
                    else "walk_forward_credit_one_lot"
                    if walk_forward_credit
                    else "probationary_one_lot"
                )
            elif debit_tier == "medium" and _medium_debit_sleeve_eligible(row):
                out.at[idx, "contracts"] = 1
                out.at[idx, "v4_execution_authority"] = "validated_medium_debit_one_lot"
            pf_text = f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf"
            out.at[idx, "v4_direct_disposition_reason"] = (
                f"V4 {'one-contract probationary Pilot' if probationary else 'direct Execute'}: Schwab reference meets target, no hard blocker, "
                f"final-structure EV ${expected_value:.2f} and PF {pf_text}, "
                + (f"{debit_tier} debit-policy tier, " if debit_tier else "")
                + (f"{credit_tier} credit-policy tier, " if credit_tier else "")
                + "defined OCO required before order entry."
            )
            out.at[idx, "trade_status_reason"] = _append_note(row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"])
        elif target_met and ev_ok and has_risk and not identification_quality_blocker:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "approved-work-limit-execution-readiness"
            out.at[idx, "decision_eligible"] = True
            out.at[idx, "decision_tier"] = "v4-validated-execution-readiness-wait"
            if execution_quote_blocker:
                reason = (
                    f"{execution_quote_blocker}; the setup remains approved because session state does not "
                    "control identification. Reprice at the next regular options session and route nothing "
                    "until a fresh executable quote still meets the target"
                )
            elif execution_catalyst_blocker:
                reason = (
                    f"{execution_catalyst_blocker}; the setup remains approved, but route nothing until "
                    "the same-session company-news and earnings check clears"
                )
            elif quote_width > quote_width_limit:
                reason = (
                    f"combination quote width {quote_width:.1%} exceeds the {quote_width_limit:.0%} execution cap; "
                    "the setup remains approved, but wait for a tighter executable market"
                )
            else:
                reason = entry_price_blocker or "a fresh executable quote is still required before entry"
            out.at[idx, "v4_direct_disposition_reason"] = f"V4 Approved Work Limit: target observed, but {reason}."
            out.at[idx, "trade_status_reason"] = _append_note(
                row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"]
            )
        elif target_met and ev_ok and has_risk:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "Scout"
            reason = (
                execution_catalyst_blocker
                or execution_quote_blocker
                or identification_quality_blocker
                or execute_quality_blocker
                or "manual confirmation keeps it to review"
            )
            out.at[idx, "v4_direct_disposition_reason"] = f"V4 Scout: target met, but {reason}."
            out.at[idx, "trade_status_reason"] = _append_note(row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"])
        elif _medium_debit_sleeve_eligible(row) and has_risk:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "debit-model-expectancy-review"
            out.at[idx, "v4_direct_disposition_reason"] = (
                "V4 Debit Review: the walk-forward model authorizes this family, but the exact quoted "
                f"structure does not clear production expectancy. {_post_pricing_expectancy_blocker(row)}"
            )
            out.at[idx, "trade_status_reason"] = _append_note(
                row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"]
            )
        elif (
            not target_met
            and ev_ok
            and has_risk
            and not identification_quality_blocker
            and (_payoff_evidence_ready(row) or _medium_debit_sleeve_eligible(row))
        ):
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "approved-work-limit-price-target"
            out.at[idx, "decision_eligible"] = True
            out.at[idx, "decision_tier"] = "v4-validated-price-wait"
            out.at[idx, "v4_direct_disposition_reason"] = (
                "V4 Approved Work Limit: every identification gate cleared, but the current Schwab "
                f"reference does not meet the validated entry limit. {entry_price_blocker or 'Do not chase.'}"
                + (
                    f" The last quote also has an execution-only blocker ({execution_quote_blocker}); "
                    "reprice during the next regular options session."
                    if execution_quote_blocker
                    else ""
                )
                + (
                    f" Execution also waits on {execution_catalyst_blocker}."
                    if execution_catalyst_blocker
                    else ""
                )
            )
            out.at[idx, "trade_status_reason"] = _append_note(
                row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"]
            )
        elif _targetable(row) or (ev_ok and has_risk and _entry_target(row) != "fresh Schwab recheck"):
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "Scout"
            reason = preprice_quality_blocker or execute_quality_blocker or _blocker_text(row)
            out.at[idx, "v4_direct_disposition_reason"] = f"V4 Scout: {reason}."
            out.at[idx, "trade_status_reason"] = _append_note(
                row.get("trade_status_reason"), out.at[idx, "v4_direct_disposition_reason"]
            )
        else:
            out.at[idx, "trade_status"] = "Research"
            out.at[idx, "trade_tier"] = "Research"
            out.at[idx, "v4_direct_disposition_reason"] = _blocker_text(row)

    pilot_rows = out[
        out["trade_status"].astype(str).eq("Execute")
        & out["trade_tier"].astype(str).str.contains("Pilot", case=False, na=False)
    ]
    if len(pilot_rows) > V4_PROBATIONARY_MAX_EXECUTES:
        ranked_pilots = sorted(
            pilot_rows.iterrows(),
            key=lambda item: _ticket_rank(item[1], "Execute"),
            reverse=True,
        )
        for idx, _ in ranked_pilots[V4_PROBATIONARY_MAX_EXECUTES:]:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "work-limit-probationary-cap"
            out.at[idx, "contracts"] = 0
            out.at[idx, "v4_direct_disposition_reason"] = (
                "V4 Work Limit: another one-contract probationary Pilot ranks higher today."
            )

    medium_debit_rows = out[
        out["trade_status"].astype(str).eq("Execute")
        & out.get("v4_execution_authority", pd.Series("", index=out.index))
        .astype(str)
        .eq("validated_medium_debit_one_lot")
    ]
    if len(medium_debit_rows) > V4_MEDIUM_DEBIT_MAX_EXECUTES:
        def medium_debit_preference(item: tuple[Any, pd.Series]) -> tuple[float, float, float, float, float]:
            _, candidate = item
            oi_status = _clean(candidate.get("oi_carryover_status")).lower()
            flow_quality = _clean(candidate.get("flow_quality")).lower()
            return (
                2.0 if oi_status == "supportive" else 1.0 if oi_status == "matched_unconfirmed" else 0.0,
                1.0 if flow_quality == "directional" else 0.0,
                safe_float(candidate.get("debit_wf_predicted_win_probability"), 0.0),
                safe_float(candidate.get("debit_wf_expected_value"), -math.inf),
                -safe_float(candidate.get("quote_width_pct"), math.inf),
            )

        ranked_medium_debits = sorted(
            medium_debit_rows.iterrows(),
            key=medium_debit_preference,
            reverse=True,
        )
        for idx, _ in ranked_medium_debits[V4_MEDIUM_DEBIT_MAX_EXECUTES:]:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "work-limit-medium-debit-cap"
            out.at[idx, "contracts"] = 0
            out.at[idx, "v4_direct_disposition_reason"] = (
                "V4 Work Limit: another validated one-contract medium-debit setup ranks higher today."
            )

    execute_rows = out[out["trade_status"].astype(str).eq("Execute")]
    for ticker, group in execute_rows.groupby(execute_rows["ticker"].astype(str).str.upper()):
        if len(group) <= 1:
            continue

        def preference(item: tuple[Any, pd.Series]) -> tuple[float, float, float]:
            _, candidate = item
            oi_status = _clean(candidate.get("oi_carryover_status")).lower()
            oi_bonus = {"supportive": 10.0, "matched_unconfirmed": 5.0, "mixed": 2.0}.get(oi_status, 0.0)
            flow_bonus = 2.0 if _clean(candidate.get("flow_quality")).lower() == "directional" else 0.0
            return (
                _ticket_rank(candidate, "Execute") + oi_bonus + flow_bonus,
                safe_float(candidate.get("edge_sample_size"), 0.0),
                -safe_float(candidate.get("quote_width_pct"), 9.0),
            )

        ranked = sorted(group.iterrows(), key=preference, reverse=True)
        keep_idx = ranked[0][0]
        for idx, _ in ranked[1:]:
            reason = f"Alternative {ticker} structure; keep only the highest-ranked same-ticker Execute and do not stack correlated variants."
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "work-limit-alternative-structure"
            out.at[idx, "v4_direct_disposition_reason"] = f"V4 Work Limit: {reason}"
            out.at[idx, "trade_status_reason"] = _append_note(out.at[idx, "trade_status_reason"], reason)
        out.at[keep_idx, "trade_status_reason"] = _append_note(
            out.at[keep_idx, "trade_status_reason"],
            f"Selected as the highest-ranked {ticker} structure; alternative same-ticker variants are Work Limit only.",
        )

    return out


V4_BROAD_INDEX_TICKERS = frozenset({"SPY", "QQQ", "IWM", "DIA", "RSP", "TQQQ", "SQQQ", "UPRO", "SPX", "NDX"})
V4_SECTOR_ALIASES = {
    "information technology": "technology",
    "technology": "technology",
    "financial services": "financials",
    "financials": "financials",
    "communication services": "communications",
    "consumer discretionary": "consumer-discretionary",
    "consumer staples": "consumer-staples",
    "health care": "healthcare",
    "healthcare": "healthcare",
    "real estate": "real-estate",
}


def _v4_credit_risk_bucket(row: pd.Series | dict[str, Any]) -> str:
    """Return the concentration bucket used for independently executable credits."""
    ticker = _clean(row.get("ticker") or row.get("Ticker")).upper()
    if ticker in V4_BROAD_INDEX_TICKERS or ticker in INDEX_FLOW_TICKERS:
        return "broad-index"

    sector = _clean(row.get("sector") or row.get("Sector")).lower()
    sector = V4_SECTOR_ALIASES.get(sector, sector.replace(" ", "-"))
    if sector and sector not in {"unknown", "nan", "none", "n/a"}:
        return f"sector:{sector}"

    if bool(row.get("index_fallback", False)):
        return f"etf:{ticker or 'unknown'}"
    return f"ticker:{ticker or 'unknown'}"


def apply_v4_credit_sleeve_cap(
    scored: pd.DataFrame,
    *,
    max_execute: int | None = None,
) -> pd.DataFrame:
    """Annotate credit correlation without suppressing valid trade identification.

    ``max_execute`` is retained only for API compatibility and is intentionally
    ignored. Actual order selection belongs to portfolio/risk review, not an
    arbitrary daily signal cap.
    """
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    if "v4_credit_risk_bucket" not in out.columns:
        out["v4_credit_risk_bucket"] = ""
    if "v4_credit_book_allocation" not in out.columns:
        out["v4_credit_book_allocation"] = ""

    execute = out[out["trade_status"].astype(str).eq("Execute")]
    credit_rows = execute[execute.apply(_is_credit, axis=1)]
    if credit_rows.empty:
        return out
    ranked = sorted(
        credit_rows.iterrows(),
        key=lambda item: (
            1 if _validated_family_evidence(item[1]) else 0,
            _ticket_rank(item[1], "Execute"),
            safe_float(item[1].get("edge_sample_size"), 0.0),
            -safe_float(item[1].get("quote_width_pct"), 9.0),
        ),
        reverse=True,
    )

    primary_by_bucket: dict[str, str] = {}
    for rank, (idx, row) in enumerate(ranked, start=1):
        bucket = _v4_credit_risk_bucket(row)
        out.at[idx, "v4_credit_risk_bucket"] = bucket
        ticker = _clean(row.get("ticker")).upper()
        if bucket not in primary_by_bucket:
            primary_by_bucket[bucket] = ticker
            note = f"confidence rank {rank}; primary {bucket} setup"
        else:
            note = (
                f"confidence rank {rank}; correlated {bucket} alternative to {primary_by_bucket[bucket]}; "
                "remains identified as Execute, but do not fill both without portfolio-risk approval"
            )
        out.at[idx, "v4_credit_book_allocation"] = note
        out.at[idx, "trade_status_reason"] = _append_note(out.at[idx, "trade_status_reason"], note)
    return out


def apply_v4_prospective_book_concentration(scored: pd.DataFrame) -> pd.DataFrame:
    """Annotate same-sector correlation without changing trade eligibility."""
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    out["proposed_book_concentration_action"] = ""
    execute = out[out["trade_status"].astype(str).eq("Execute")]
    if execute.empty or "sector" not in execute.columns:
        return out
    sectors = execute["sector"].astype(str).str.strip()
    for sector, group in execute.groupby(sectors):
        if not sector or sector.lower() in {"unknown", "nan", "none"} or len(group) <= 1:
            continue

        def preference(item: tuple[Any, pd.Series]) -> tuple[float, float, float]:
            _, candidate = item
            return (
                _ticket_rank(candidate, "Execute"),
                safe_float(candidate.get("edge_sample_size"), 0.0),
                -safe_float(candidate.get("quote_width_pct"), 9.0),
            )

        ranked = sorted(group.iterrows(), key=preference, reverse=True)
        keep_idx = ranked[0][0]
        out.at[keep_idx, "proposed_book_concentration_action"] = f"primary {sector} Execute"
        for idx, _ in ranked[1:]:
            reason = (
                f"Same-sector correlation warning: a higher-ranked {sector} setup is also Execute; "
                "trade remains identified, but simultaneous fills require portfolio-risk approval."
            )
            out.at[idx, "trade_status_reason"] = _append_note(out.at[idx, "trade_status_reason"], reason)
            out.at[idx, "proposed_book_concentration_action"] = "correlated Execute; portfolio review required"
    return out


def _status_label(disposition: str) -> str:
    return {
        "Execute": "🟣 ENTER NOW",
        "Swing Target / Work Limit": "🟢 WORK LIMIT",
        "Scout": "🟡 REVIEW / Scout",
        "Portfolio Repair": "🟠 PORTFOLIO / Repair",
        "Wheel/Cash": "🔵 WHEEL / Cash",
        "Research": "🟡 RESEARCH / Review",
        "Avoid": "🔴 REJECT / Avoid",
    }.get(disposition, disposition)


def build_v4_opportunity_board(scored: pd.DataFrame, *, top_flow: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "Modeled win rate",
        "Win-rate basis",
        "Confidence evidence",
        "Payoff evidence",
        "Per-ticket replay edge",
        "Post-pricing EV / PF",
        "Expected value source",
        "Edge sample size / win rate / avg P/L",
        "Required confirmation",
        "Why Execute, Scout, Research, or Avoid",
    ]
    if scored is None or scored.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in scored.iterrows():
        targetable = _targetable(row)
        disposition = _disposition(row, targetable=targetable)
        sample = safe_float(row.get("edge_sample_size"))
        raw_win = safe_float(row.get("edge_win_rate"))
        win = safe_float(row.get("edge_effective_win_rate"), raw_win)
        match_level = _clean(row.get("edge_match_level")) or "unavailable"
        avg = safe_float(row.get("edge_avg_pnl"))
        modeled_win = _reported_win_rate(row)
        post_ev, post_pf, _, _ = _post_pricing_expectancy(row)
        post_pf_text = f"{post_pf:.2f}" if math.isfinite(post_pf) else "inf" if post_pf == math.inf else "unavailable"
        edge_summary = (
            f"n={int(sample)} / smoothed historical win={win:.0%} / avg=${avg:.2f} / "
            f"calibration={_clean(row.get('confidence_calibration_status')) or 'unavailable'}"
            if math.isfinite(sample) and sample > 0 and math.isfinite(win) and math.isfinite(avg)
            else "exact/ticker match unavailable; validated route evidence reported separately"
            if _payoff_model_ready(row)
            else "exact/ticker match unavailable"
        )
        rows.append(
            {
                "Lane": _lane(row, disposition),
                "Status": _status_label(disposition),
                "Ticker": _clean(row.get("ticker")).upper(),
                "Trade": _trade_legs(row),
                "Expiry": _clean(row.get("expiry")),
                "Entry limit": _expectancy_safe_entry_target(row),
                "Live mid/natural": _mid_natural(row),
                "Max profit": _money(row.get("max_profit")),
                "Max loss": _money(_max_loss_value(row)),
                "Target profit": _money(_target_profit_value(row)),
                "Modeled win rate": f"{modeled_win:.0%}" if math.isfinite(modeled_win) else "unavailable",
                "Win-rate basis": _reported_win_rate_basis(row),
                "Confidence evidence": _confidence_evidence_text(row),
                "Payoff evidence": _payoff_evidence_text(row),
                "Per-ticket replay edge": _edge_evidence_text(row),
                "Post-pricing EV / PF": (
                    f"EV={_money(post_ev)}; PF={post_pf_text}"
                    if math.isfinite(post_ev) and not math.isnan(post_pf)
                    else "unavailable"
                ),
                "Expected value source": (
                    f"Schwab {_clean(row.get('live_status'))}; "
                    f"edge {_clean(row.get('edge_verdict') or row.get('replay_ev_verdict'))}; "
                    f"match {match_level}"
                ),
                "Edge sample size / win rate / avg P/L": edge_summary,
                "Required confirmation": _blocker_text(row),
                "Why Execute, Scout, Research, or Avoid": _clean(row.get("v4_direct_disposition_reason")) or _why_review(row, top_flow),
            }
        )
    board = pd.DataFrame(rows, columns=columns)
    if board.empty:
        return board
    rank = board["Status"].astype(str).map(lambda s: 6 if "Execute" in s else 5 if "Swing Target" in s else 4 if "Scout" in s else 3 if "Wheel" in s else 2 if "Research" in s else 1).fillna(0)
    board["_rank"] = rank
    board = board.sort_values(["_rank", "Ticker"], ascending=[False, True]).drop(columns=["_rank"]).reset_index(drop=True)
    return board


def build_strategy_generation_coverage(
    *,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    if registry is None or registry.empty:
        return pd.DataFrame()
    registry_columns = [
        "strategy_key",
        "display_name",
        "category",
        "live_builder",
        "historical_scope",
        "pipeline_status",
        "strategy_validation_status",
        "strategy_validation_scope",
        "strategy_validation_scope_value",
        "strategy_validation_clustered_pf_p05",
        "strategy_validation_holm_p",
        "execution_authorized",
        "status_reason",
    ]
    coverage = registry.copy()
    for column in registry_columns:
        if column not in coverage.columns:
            coverage[column] = "NOT_EVALUATED" if column == "strategy_validation_status" else ""
    coverage = coverage[registry_columns].copy()

    def summarize(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["strategy_key"])
        work = frame.copy()
        work["strategy_key"] = work.apply(strategy_key_for_row, axis=1)
        grouped = work.groupby("strategy_key", dropna=False)
        aggregations: dict[str, tuple[str, str]] = {
            f"{prefix}_rows": ("strategy_key", "size"),
            f"{prefix}_tickers": ("ticker", "nunique"),
        }
        if "sector" in work.columns:
            aggregations[f"{prefix}_sectors"] = ("sector", "nunique")
        summary = grouped.agg(**aggregations).reset_index()
        if prefix == "constructed":
            pass_counts = (
                work[work.get("live_status", pd.Series("", index=work.index)).astype(str).eq("PASS")]
                .groupby("strategy_key")
                .size()
                .rename("live_pass_rows")
            )
            execute_counts = (
                work[work.get("trade_status", pd.Series("", index=work.index)).astype(str).eq("Execute")]
                .groupby("strategy_key")
                .size()
                .rename("execute_rows")
            )
            examples = grouped["ticker"].agg(
                lambda values: ", ".join(sorted(set(values.astype(str).str.upper()))[:5])
            ).rename("example_tickers")
            summary = summary.merge(pass_counts, on="strategy_key", how="left")
            summary = summary.merge(execute_counts, on="strategy_key", how="left")
            summary = summary.merge(examples, on="strategy_key", how="left")
        return summary

    coverage = coverage.merge(summarize(candidates, "seed"), on="strategy_key", how="left")
    coverage = coverage.merge(summarize(scored, "constructed"), on="strategy_key", how="left")
    for column in [
        "seed_rows",
        "seed_tickers",
        "seed_sectors",
        "constructed_rows",
        "constructed_tickers",
        "constructed_sectors",
        "live_pass_rows",
        "execute_rows",
    ]:
        coverage[column] = pd.to_numeric(coverage.get(column), errors="coerce").fillna(0).astype(int)
    coverage["generation_status"] = "NOT_SEEDED"
    coverage.loc[coverage["seed_rows"].gt(0), "generation_status"] = "SEEDED_NO_LIVE_STRUCTURE"
    coverage.loc[coverage["live_pass_rows"].gt(0), "generation_status"] = "LIVE_CONSTRUCTED"
    coverage["example_tickers"] = coverage.get("example_tickers", pd.Series("", index=coverage.index)).fillna("")
    return coverage.sort_values(["generation_status", "category", "display_name"]).reset_index(drop=True)


def write_v4_outputs(
    *,
    out_dir: Path,
    base_dir: Path,
    asof: dt.date,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    board: pd.DataFrame,
    top_flow: pd.DataFrame,
    flow_velocity: pd.DataFrame,
    correlation: pd.DataFrame,
    macro: pd.DataFrame,
    confirmation: pd.DataFrame,
    data_quality: dict[str, Any],
    portfolio: dict[str, Any],
    regime: dict[str, Any],
    regime_context: dict[str, Any],
    recent_performance: dict[str, Any],
    live_outcomes: dict[str, Any],
    loss_review: dict[str, Any],
    liquidity_summary: dict[str, Any],
    uw_source_status: dict[str, Any] | None = None,
    run_mode: str = RUN_MODE_V4,
    sector_rotation: pd.DataFrame | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _normalize_v4_dataframe(candidates)
    scored = _normalize_v4_dataframe(scored)
    board = _normalize_v4_dataframe(board)
    top_flow = _normalize_v4_dataframe(top_flow)
    flow_velocity = _normalize_v4_dataframe(flow_velocity)
    correlation = _normalize_v4_dataframe(correlation)
    macro = _normalize_v4_dataframe(macro)
    confirmation = _normalize_v4_dataframe(confirmation)
    sector_rotation = _normalize_v4_dataframe(sector_rotation if sector_rotation is not None else pd.DataFrame())
    scored = apply_sector_rotation_context(scored, sector_rotation)
    data_quality = _v4_text(data_quality)
    portfolio = _v4_text(portfolio)
    regime = _v4_text(regime)
    regime_context = _v4_text(regime_context)
    recent_performance = _v4_text(recent_performance)
    live_outcomes = _v4_text(live_outcomes)
    loss_review = _v4_text(loss_review)
    liquidity_summary = _v4_text(liquidity_summary)
    uw_source_status = _v4_text(uw_source_status or {})
    latest_asof = latest_dated_folder(base_dir.parent)
    note = live_planning_validation_note(asof, latest_asof)
    if note:
        regime["validation_note"] = note

    outcome_ledger = load_recommendation_ledgers(base_dir.parent / "out")
    safety_calibration = build_v4_safety_calibration(
        scored=scored,
        outcome_ledger=outcome_ledger,
        asof=asof,
        lookback_days=14,
    )
    scored = apply_v4_safety_calibration(scored, safety_calibration)
    confidence_predictions, confidence_calibration = build_default_walk_forward_calibration(asof=asof)
    scored = apply_confidence_calibration(scored, confidence_calibration)
    payoff_summary, payoff_groups, payoff_walk_forward = build_default_payoff_calibration(asof=asof)
    scored = apply_payoff_calibration(scored, payoff_groups)
    from .daily_shadow_books import apply_live_debit_execution_model

    scored, debit_execution_summary = apply_live_debit_execution_model(
        scored,
        root=base_dir.parent,
        asof=asof,
        allow_execution_authority=(run_mode == RUN_MODE_V4),
    )
    symbol_credit_summary, symbol_credit_evidence = build_symbol_credit_calibration(
        base_dir.parent,
        asof,
        assessor=assess_credit_spread,
        screener_loader=load_stock_screener,
        history_path=DEFAULT_EDGE_HISTORY_PATH,
        credit_policy_version=CREDIT_POLICY_VERSION,
    )
    scored = apply_symbol_credit_calibration(scored, symbol_credit_summary, assessor=assess_credit_spread)
    symbol_credit_paths = write_symbol_credit_calibration_outputs(
        out_dir=out_dir,
        asof=asof,
        summary=symbol_credit_summary,
        evidence=symbol_credit_evidence,
    )
    walk_forward_credit_summary, walk_forward_credit_evidence, walk_forward_credit_model = (
        build_walk_forward_credit_model(asof=asof)
    )
    scored = apply_walk_forward_credit_model(scored, walk_forward_credit_summary, walk_forward_credit_model)
    walk_forward_credit_paths = write_walk_forward_credit_outputs(
        out_dir=out_dir,
        asof=asof,
        summary=walk_forward_credit_summary,
        evidence=walk_forward_credit_evidence,
        qualified=scored,
    )
    payoff_paths = write_payoff_calibration_outputs(
        out_dir=out_dir,
        asof=asof,
        summary=payoff_summary,
        groups=payoff_groups,
        walk_forward=payoff_walk_forward,
    )
    strategy_validation_source_path = base_dir.parent / "out" / "sector_strategy_validation_v3.csv"
    strategy_validation = (
        pd.read_csv(strategy_validation_source_path, low_memory=False)
        if strategy_validation_source_path.exists()
        else pd.DataFrame()
    )
    strategy_registry = build_strategy_registry(
        payoff_summary=payoff_summary,
        payoff_groups=payoff_groups,
        confidence_summary=confidence_calibration,
        strategy_validation=strategy_validation,
    )
    execution_evidence_integrity = build_execution_evidence_integrity(scored)
    scored = apply_v4_professional_dispositions(scored, asof=asof)
    scored = apply_strategy_registry_gate(scored, strategy_registry)
    scored = apply_v4_credit_sleeve_cap(scored)
    scored = apply_v4_prospective_book_concentration(scored)
    execution_reachability = build_execution_reachability_audit(scored)
    strategy_generation_coverage = build_strategy_generation_coverage(
        candidates=candidates,
        scored=scored,
        registry=strategy_registry,
    )
    goal_shadow, goal_shadow_paths, goal_shadow_summary = write_goal_shadow_outputs(
        scored,
        out_dir=out_dir,
        asof=asof,
        source_scored_file=str(_artifact_path(out_dir, "scored_reference", asof)),
        root=base_dir.parent,
        resolve_through_date=asof,
    )
    board = build_v4_opportunity_board(scored, top_flow=top_flow)
    tickets = build_v4_swing_target_tickets(scored=scored, board=board, regime=regime, top_flow=top_flow)
    tickets, risk_cap_audit = apply_v4_risk_cap(tickets, portfolio)
    tickets = _attach_schwab_gex_context(tickets, out_dir / "schwab_chains")
    public_board = _public_opportunity_board(board)
    raw_universe = build_raw_universe(top_flow=top_flow, scored=scored, candidates=candidates, portfolio=portfolio)
    dispositions = build_candidate_disposition(candidates=candidates, scored=scored, top_flow=top_flow, tickets=tickets)
    attempts = build_construction_attempts(scored=scored, top_flow=top_flow, tickets=tickets, portfolio=portfolio)
    suppression = build_suppression_audit(dispositions)
    no_miss = build_no_miss_audit(top_flow=top_flow, scored=scored, dispositions=dispositions, attempts=attempts, tickets=tickets)
    secondary_sweep = build_secondary_liquidity_sweep(
        candidates=candidates,
        scored=scored,
        top_flow=top_flow,
        flow_velocity=flow_velocity,
        correlation=correlation,
    )
    generation_audit = build_generation_audit(
        top_flow=top_flow,
        scored=scored,
        dispositions=dispositions,
        tickets=tickets,
        board=board,
        secondary_sweep=secondary_sweep,
    )
    portfolio_repair = build_portfolio_repair(portfolio)
    target_model = build_v4_target_model(
        asof=asof,
        tickets=tickets,
        portfolio=portfolio,
        monthly_profit_target=args.monthly_profit_target,
        month_to_date_realized_pnl=args.month_to_date_realized_pnl,
        open_unrealized_pnl=args.open_unrealized_pnl,
        risk_budget=args.risk_budget,
    )

    paths = {
        "raw_universe": _artifact_path(out_dir, "raw_universe", asof),
        "candidate_disposition": _artifact_path(out_dir, "candidate_disposition", asof),
        "swing_target_tickets": _artifact_path(out_dir, "swing_target_tickets", asof),
        "suppression_audit": _artifact_path(out_dir, "suppression_audit", asof),
        "construction_attempts": _artifact_path(out_dir, "construction_attempts", asof),
        "no_miss_audit": _artifact_path(out_dir, "no_miss_audit", asof),
        "generation_audit": _artifact_path(out_dir, "candidate_generation_audit", asof),
        "safety_calibration": _artifact_path(out_dir, "safety_calibration", asof),
        "confidence_calibration_predictions": _artifact_path(out_dir, "confidence_calibration_predictions", asof),
        "confidence_calibration_summary": out_dir / f"codexdaily_v4_confidence_calibration_summary_{asof}.json",
        "execution_reachability": _artifact_path(out_dir, "execution_reachability", asof, "json"),
        "risk_cap_audit": _artifact_path(out_dir, "risk_cap_audit", asof),
        "secondary_liquidity_sweep": _artifact_path(out_dir, "secondary_liquidity_sweep", asof),
        "portfolio_repair": _artifact_path(out_dir, "portfolio_repair", asof),
        "opportunity_board": _artifact_path(out_dir, "opportunity_board", asof),
        "scored_reference": _artifact_path(out_dir, "scored_reference", asof),
        "macro_event_gates": _artifact_path(out_dir, "macro_event_gates", asof),
        "confirmation_evidence": _artifact_path(out_dir, "confirmation_evidence", asof),
        "strategy_registry": _artifact_path(out_dir, "strategy_registry", asof),
        "strategy_validation": _artifact_path(out_dir, "strategy_validation", asof),
        "strategy_generation_coverage": _artifact_path(out_dir, "strategy_generation_coverage", asof),
        "sector_rotation": _artifact_path(out_dir, "sector_rotation", asof),
    }
    raw_universe.to_csv(paths["raw_universe"], index=False)
    dispositions.to_csv(paths["candidate_disposition"], index=False)
    tickets.to_csv(paths["swing_target_tickets"], index=False)
    suppression.to_csv(paths["suppression_audit"], index=False)
    attempts.to_csv(paths["construction_attempts"], index=False)
    no_miss.to_csv(paths["no_miss_audit"], index=False)
    generation_audit.to_csv(paths["generation_audit"], index=False)
    safety_calibration.to_csv(paths["safety_calibration"], index=False)
    confidence_predictions.to_csv(paths["confidence_calibration_predictions"], index=False)
    paths["confidence_calibration_summary"].write_text(
        json.dumps(confidence_calibration, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    paths["execution_reachability"].write_text(
        json.dumps(execution_reachability, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    risk_cap_audit.to_csv(paths["risk_cap_audit"], index=False)
    secondary_sweep.to_csv(paths["secondary_liquidity_sweep"], index=False)
    portfolio_repair.to_csv(paths["portfolio_repair"], index=False)
    public_board.to_csv(paths["opportunity_board"], index=False)
    scored.to_csv(paths["scored_reference"], index=False)
    macro.to_csv(paths["macro_event_gates"], index=False)
    confirmation.to_csv(paths["confirmation_evidence"], index=False)
    strategy_registry.to_csv(paths["strategy_registry"], index=False)
    strategy_validation.to_csv(paths["strategy_validation"], index=False)
    strategy_generation_coverage.to_csv(paths["strategy_generation_coverage"], index=False)
    sector_rotation.to_csv(paths["sector_rotation"], index=False)

    no_miss_md = out_dir / f"codexdaily_v4_no_miss_audit_{asof}.md"
    no_miss_md.write_text(
        "\n".join(
            [
                f"# {PIPELINE_NAME_V4} No-Miss Audit - {asof}",
                "",
                "## Top-Flow Disposition",
                "",
                _markdown_table(
                    no_miss,
                    [
                        "top_flow_rank",
                        "ticker",
                        "candidate_generated",
                        "constructions_attempted",
                        "best_construction",
                        "final_disposition",
                        "shown_as_target_ticket",
                        "if_not_targetable_exact_reason",
                        "hard_or_non_hard_blocker",
                    ],
                    max_rows=50,
                ),
                "",
                "## Candidate Generation Checks",
                "",
                _markdown_table(generation_audit, ["audit_item", "audit_status", "evidence"]),
                "",
                "## Safety Calibration",
                "",
                _markdown_table(
                    safety_calibration,
                    [
                        "setup_family_key",
                        "outcome_count",
                        "loss_count",
                        "loss_rate",
                        "expected_value",
                        "strategy_slump_muted",
                        "mute_until",
                        "shadow_backtest_status",
                    ],
                    max_rows=100,
                ),
                "",
                "## Risk Cap Audit",
                "",
                _markdown_table(
                    risk_cap_audit,
                    [
                        "ticker",
                        "final disposition before",
                        "final disposition after",
                        "two_percent_loss_cap",
                        "max_loss_before",
                        "max_loss_after",
                        "risk_cap_action",
                        "risk_capped",
                    ],
                    max_rows=100,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    suppression_md = out_dir / f"codexdaily_v4_suppression_audit_{asof}.md"
    suppression_md.write_text(
        "\n".join(
            [
                f"# {PIPELINE_NAME_V4} Suppression Audit - {asof}",
                "",
                _markdown_table(
                    suppression,
                    [
                        "ticker",
                        "thesis",
                        "strategy",
                        "human_readable_trade_legs",
                        "final_disposition",
                        "exact_reason",
                        "hard_or_non_hard_blocker",
                        "targetable_trade_hidden_by_price_miss",
                    ],
                    max_rows=100,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    counts = {
        "execute": int(tickets["final disposition"].eq("Execute").sum()) if not tickets.empty else 0,
        "swing_target_work_limit": int(tickets["final disposition"].eq("Swing Target / Work Limit").sum()) if not tickets.empty else 0,
        "scout": int(tickets["final disposition"].eq("Scout").sum()) if not tickets.empty else 0,
        "wheel_cash": int(tickets["final disposition"].eq("Wheel/Cash").sum()) if not tickets.empty else 0,
        "research": int(dispositions["final_disposition"].eq("Research").sum()) if not dispositions.empty else 0,
        "avoid": int(dispositions["final_disposition"].eq("Avoid").sum()) if not dispositions.empty else 0,
    }
    lane_coverage = tickets["lane"].value_counts().to_dict() if not tickets.empty else {}
    artifacts = {name: str(path) for name, path in paths.items()}
    artifacts.update(payoff_paths)
    artifacts.update(symbol_credit_paths)
    artifacts.update(walk_forward_credit_paths)
    artifacts.update(goal_shadow_paths)
    artifacts.update(
        {
            "report": str(out_dir / f"codexdaily_v4_report_{asof}.md"),
            "manifest": str(out_dir / f"codexdaily_v4_manifest_{asof}.json"),
            "no_miss_audit_markdown": str(no_miss_md),
            "suppression_audit_markdown": str(suppression_md),
        }
    )
    schwab_status = _schwab_status(data_quality)
    regime_label = f"{regime.get('trend', 'unknown')}; vol={regime.get('volatility', 'unknown')}; flow={regime.get('flow', 'unknown')}; VIX={regime.get('vix_proxy', 'n/a')}"
    target_profit = target_model["swing_target_profit_potential_if_filled"]
    target_risk = target_model["max_risk_if_all_target_tickets_fill"]
    fill_adjusted = target_model["realistic_fill_adjusted_target_potential"]
    target_status = _target_status_panel(target_model, counts)
    execute_tickets = tickets[tickets["final disposition"].eq("Execute")]
    approved_tickets = tickets[
        tickets["final disposition"].eq("Swing Target / Work Limit")
    ]
    scout_tickets = tickets[tickets["final disposition"].eq("Scout")]
    compact_execute_tickets = _compact_ticket_table(execute_tickets)
    compact_approved_tickets = _compact_ticket_table(approved_tickets)
    compact_scout_tickets = _compact_ticket_table(scout_tickets)
    high_confidence_ticket_count = int(
        tickets["confidence rating"].fillna("").astype(str).str.contains("HIGH", case=False).sum()
    )
    compact_decision_candidates = _compact_decision_candidate_table(scored)
    compact_portfolio_repair = _compact_portfolio_repair_table(portfolio_repair)
    compact_no_miss = _compact_no_miss_table(no_miss)
    compact_board = _compact_opportunity_board(public_board)
    muted_families = int(safety_calibration["strategy_slump_muted"].astype(bool).sum()) if not safety_calibration.empty else 0
    negative_shadow_ev = int(safety_calibration["shadow_backtest_status"].astype(str).eq("negative_ev").sum()) if not safety_calibration.empty else 0
    strategy_status_counts = strategy_registry["pipeline_status"].value_counts().to_dict()
    emerging_sectors = (
        sorted(sector_rotation.loc[sector_rotation["sector_state"].eq("emerging"), "sector"].astype(str).unique())
        if not sector_rotation.empty and {"sector_state", "sector"}.issubset(sector_rotation.columns)
        else []
    )
    risk_capped_count = int(risk_cap_audit["risk_capped"].astype(bool).sum()) if not risk_cap_audit.empty else 0
    secondary_triggered = bool(secondary_sweep["triggered"].astype(bool).any()) if not secondary_sweep.empty else False
    missed_opportunity = build_missed_opportunity_audit(outcome_ledger)
    outcome_learning = {
        "status": "ok" if not missed_opportunity.empty else "no_later_working_rejections_recorded",
        "rows": int(len(missed_opportunity)),
        "classifications": missed_opportunity["classification"].value_counts().to_dict() if not missed_opportunity.empty else {},
    }
    manifest = {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V4,
        "version_lock": pipeline_version_record("v4"),
        "run_mode": RUN_MODE_V4,
        "asof": str(asof),
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "data_quality": data_quality,
        "uw_source_status": uw_source_status,
        "schwab_status": schwab_status,
        "portfolio_status": (portfolio or {}).get("status", "not_checked"),
        "market_regime": regime,
        "payoff_calibration": payoff_summary,
        "symbol_credit_calibration": symbol_credit_summary,
        "walk_forward_credit_calibration": walk_forward_credit_summary,
        "goal_shadow": goal_shadow_summary,
        "execution_evidence_integrity": execution_evidence_integrity,
        "execution_reachability": execution_reachability,
        "swing_target_ticket_count": int(len(tickets)),
        "high_confidence_ticket_count": high_confidence_ticket_count,
        "decision_lane_candidate_count": int(len(compact_decision_candidates)),
        "opportunity_counts": counts,
        "lane_coverage": lane_coverage,
        "target_model": target_model,
        "visible_signal_policy": {
            "no_miss_reporting": True,
            "max_final_trades_arg": int(getattr(args, "max_final_trades", 0) or 0),
            "active_execute_cap": None,
            "aggregate_risk_budget_arg": float(getattr(args, "risk_budget", 0.0) or 0.0),
            "aggregate_risk_budget_applied": bool(target_model.get("aggregate_risk_budget_applied")),
            "cap_reason": (
                "--max-final-trades is ignored in Codex Daily V4; valid Execute/Swing Target rows stay visible. "
                "Default aggregate risk budget is disabled; the per-ticket 2% account-loss level is advisory. "
                "Target limits, portfolio context, and disposition still affect execution quality."
            ),
        },
        "no_miss_audit": {
            "top_flow_tickers_audited": int(len(no_miss)),
            "candidate_disposition_rows": int(len(dispositions)),
            "suppressed_rows": int(len(suppression)),
            "targetable_price_misses_hidden": int(suppression.get("targetable_trade_hidden_by_price_miss", pd.Series(dtype=bool)).astype(bool).sum()) if not suppression.empty else 0,
            "generation_checks": generation_audit.to_dict(orient="records"),
        },
        "safety_calibration": {
            "rolling_window_days": 14,
            "strategy_slump_muted_families": muted_families,
            "negative_shadow_ev_families": negative_shadow_ev,
            "risk_capped_tickets": risk_capped_count,
            "risk_cap_pct_account_value": 0.02,
            "secondary_liquidity_sweep_triggered": secondary_triggered,
        },
        "confidence_calibration": confidence_calibration,
        "debit_execution_lane": debit_execution_summary,
        "strategy_registry": {
            "family_count": int(len(strategy_registry)),
            "live_builder_count": int(strategy_registry["live_builder"].astype(bool).sum()),
            "seeded_family_count": int(strategy_generation_coverage["seed_rows"].gt(0).sum()),
            "constructed_family_count": int(strategy_generation_coverage["live_pass_rows"].gt(0).sum()),
            "unseeded_families": strategy_generation_coverage.loc[
                strategy_generation_coverage["seed_rows"].eq(0), "strategy_key"
            ].astype(str).tolist(),
            "unconstructed_families": strategy_generation_coverage.loc[
                strategy_generation_coverage["live_pass_rows"].eq(0), "strategy_key"
            ].astype(str).tolist(),
            "status_counts": strategy_status_counts,
            "execution_authorized_count": int(strategy_registry["execution_authorized"].astype(bool).sum()),
            "probationary_execution_authorized_count": int(
                strategy_registry["probationary_execution_authorized"].astype(bool).sum()
            ),
        },
        "sector_rotation": {
            "authority": "prospective_context_only",
            "sector_count": int(sector_rotation["sector"].nunique()) if not sector_rotation.empty and "sector" in sector_rotation.columns else 0,
            "emerging_sectors": emerging_sectors,
        },
        "outcome_learning": outcome_learning,
        "artifacts": artifacts,
        "scoring_source": "direct_v4_eod_pipeline",
        "status": "blocked" if data_quality.get("status") == "critical" else "ok",
    }
    manifest_path = out_dir / f"codexdaily_v4_manifest_{asof}.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = artifacts["report"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    report_path = out_dir / f"codexdaily_v4_report_{asof}.md"
    report_lines = [
        f"# {PIPELINE_NAME_V4} Report - {asof}",
        "",
        "## Action Board",
        "",
        "This is the trading screen. **No order was placed.** Recheck the live Schwab quote, earnings, portfolio overlap, and OCO before entry. Multiple bullish put-credit rows are one correlated strategy sleeve, not independent diversification.",
        "",
        f"### 🟣 Enter Now / Execute — {len(compact_execute_tickets)}",
        "",
        _markdown_table(_readable_action_ticket_table(execute_tickets)),
        "",
        "#### Enter Now ticket details",
        *_readable_ticket_detail_lines(execute_tickets),
        "",
        f"### 🟢 Work Limit / Approved — {len(compact_approved_tickets)}",
        "",
        "These rows cleared setup-identification gates. Price, quote, or same-session evidence may still block entry; follow each row's stated condition.",
        "",
        _markdown_table(_readable_action_ticket_table(approved_tickets)),
        "",
        "#### Work Limit ticket details",
        *_readable_ticket_detail_lines(approved_tickets),
        "",
        f"### 🟡 Review / Scout — {len(compact_scout_tickets)}",
        "",
        "Review rows are not trades and have no production execution authority.",
        "",
        _markdown_table(_readable_action_ticket_table(scout_tickets)),
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V4} |",
        f"| Version | {PIPELINE_VERSION_V4} |",
        f"| Version lock | locked {pipeline_version_record('v4')['locked_on']}; rollback chain retained through v4.0 |",
        f"| Run mode | {RUN_MODE_V4} |",
        f"| Data quality | {data_quality.get('status', 'unknown')} |",
        f"| Schwab status | {schwab_status} |",
        f"| Symbol-trend credit lane | {symbol_credit_summary.get('status', 'FAIL')}; n={symbol_credit_summary.get('sample_size', 0)}, OOS n={symbol_credit_summary.get('oos_sample_size', 0)}, 10%-stress PF={symbol_credit_summary.get('stress_profit_factor_10pct', 'n/a')} |",
        f"| Directional credit Medium lane | {walk_forward_credit_summary.get('directional_credit_lane', {}).get('status', 'FAIL')}; "
        f"execution-policy n={walk_forward_credit_summary.get('directional_credit_lane', {}).get('execution_sample_size', 0)}, "
        f"95% floor={safe_float(walk_forward_credit_summary.get('directional_credit_lane', {}).get('execution_wilson_lower_bound'), 0.0):.0%}, "
        f"10%-stress PF={walk_forward_credit_summary.get('directional_credit_lane', {}).get('execution_stress_profit_factor_10pct', 'n/a')}; High unavailable |",
        f"| Bull-call debit pilot | {'ACTIVE' if debit_execution_summary.get('execution_authority_enabled') else 'BLOCKED'}; "
        f"live-guard n={debit_execution_summary.get('live_guard_rows', 0)}; "
        f"production candidates={debit_execution_summary.get('production_candidate_rows', 0)}; one contract maximum |",
        f"| Portfolio status | {(portfolio or {}).get('status', 'not_checked')} |",
        f"| Market regime | {regime_label} |",
        f"| Swing target ticket count | {len(tickets)} |",
        f"| Decision-lane candidate count | {len(compact_decision_candidates)} |",
        f"| Execute count | {counts['execute']} |",
        f"| Evidence-authorized rows | {execution_reachability.get('evidence_authorized_rows', 0)} |",
        f"| Approved rows independent of current fill | {execution_reachability.get('approved_target_rows', 0)} |",
        f"| Current entry target met before same-ticker selection | {execution_reachability.get('entry_ready_before_same_ticker_selection', 0)} |",
        f"| Scout count | {counts['scout']} |",
        "| Visible signal cap | none; --max-final-trades is ignored by V4 no-miss policy |",
        f"| Aggregate risk budget | {'applied' if target_model['aggregate_risk_budget_applied'] else 'not configured'} |",
        f"| $10k/month status | {target_model['target_feasibility']}: {_clean(target_status.iloc[1]['Answer']).lower()} |",
        f"| Visible expected potential | ${target_model['risk_bounded_fill_adjusted_target_potential']:,.2f} |",
        f"| Immediate Execute profit potential | ${target_model['execute_profit_potential']:,.2f} |",
        f"| Safety calibration | muted families {muted_families}; negative shadow-EV families {negative_shadow_ev}; risk-capped tickets {risk_capped_count} |",
        f"| Confidence calibration | {confidence_calibration.get('status', 'UNAVAILABLE')}; "
        f"walk-forward n={confidence_calibration.get('prediction_count', 0)}; "
        f"High-label capability={'yes' if confidence_calibration.get('high_confidence_available') else 'no'} |",
        f"| High-confidence ticket count | {high_confidence_ticket_count}; High requires both a validated payoff route and High-ready confidence evidence; current directional-credit and one-lot debit authorities are Medium-only |",
        f"| Strategy registry | {len(strategy_registry)} families; "
        f"live-builders={int(strategy_registry['live_builder'].astype(bool).sum())}; "
        f"production={strategy_status_counts.get('PRODUCTION', 0)}; "
        f"probationary={strategy_status_counts.get('PROBATIONARY', 0)}; "
        f"prospective={strategy_status_counts.get('PROSPECTIVE', 0)}; "
        f"research={strategy_status_counts.get('RESEARCH_ONLY', 0)}; "
        f"data-gap={strategy_status_counts.get('UNTESTED_DATA_GAP', 0)} |",
        f"| Strategy generation | seeded={int(strategy_generation_coverage['seed_rows'].gt(0).sum())}/{len(strategy_registry)}; "
        f"live-constructed={int(strategy_generation_coverage['live_pass_rows'].gt(0).sum())}/{len(strategy_registry)} |",
        f"| Sector rotation | {len(emerging_sectors)} emerging: {', '.join(emerging_sectors) if emerging_sectors else 'none'}; prospective context only |",
        f"| Secondary Liquidity Sweep | {'triggered' if secondary_triggered else 'not triggered'} |",
        f"| Goal shadow | {len(goal_shadow)} candidate(s); shadow-only; no order placement |",
        "",
        "## Market Insight For Tomorrow",
        "",
        *_market_insight_lines(regime_context, target_model, tickets),
        "",
        "## Execution Reachability Audit",
        "",
        "Identification is independent of whether the market is currently open. Enter/Execute still requires a usable regular-session option quote, the validated limit, defined risk, and a fresh pre-order check.",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"Gate": "Evidence authorized", "Rows": execution_reachability.get("evidence_authorized_rows", 0)},
                    {"Gate": "Hard risk clear", "Rows": execution_reachability.get("hard_risk_clear_rows", 0)},
                    {"Gate": "Non-price quality clear", "Rows": execution_reachability.get("non_price_quality_clear_rows", 0)},
                    {"Gate": "Approved target", "Rows": execution_reachability.get("approved_target_rows", 0)},
                    {"Gate": "Current target + quote ready", "Rows": execution_reachability.get("entry_ready_before_same_ticker_selection", 0)},
                    {"Gate": "Final Execute", "Rows": execution_reachability.get("final_execute_rows", 0)},
                    {"Gate": "Price wait", "Rows": execution_reachability.get("price_wait_rows", 0)},
                    {"Gate": "Quote-width wait", "Rows": execution_reachability.get("quote_wait_rows", 0)},
                ]
            )
        ),
        "",
        f"Reachability status: **{execution_reachability.get('status', 'UNKNOWN')}** — {execution_reachability.get('reason', '')}",
        "",
        "## Strategy Coverage Registry",
        "",
        "Registry, generation, construction, evidence, and execution authority are separate. A family is counted as live only when a scored row contains a successfully constructed current-chain structure.",
        "",
        _markdown_table(
            strategy_registry,
            [
                "display_name",
                "category",
                "outlook",
                "risk_profile",
                "live_builder",
                "pipeline_status",
                "payoff_evidence_status",
                "confidence_evidence_status",
                "strategy_validation_status",
                "strategy_validation_scope",
                "strategy_validation_scope_value",
                "strategy_validation_clustered_pf_p05",
                "strategy_validation_holm_p",
                "execution_authorized",
                "probationary_execution_authorized",
                "status_reason",
            ],
            max_rows=40,
        ),
        "",
        "## Generated Strategy Coverage",
        "",
        _markdown_table(
            strategy_generation_coverage,
            [
                "display_name",
                "category",
                "pipeline_status",
                "historical_scope",
                "strategy_validation_status",
                "strategy_validation_scope",
                "strategy_validation_scope_value",
                "strategy_validation_clustered_pf_p05",
                "strategy_validation_holm_p",
                "seed_rows",
                "seed_tickers",
                "seed_sectors",
                "constructed_rows",
                "constructed_tickers",
                "constructed_sectors",
                "live_pass_rows",
                "generation_status",
                "execution_authorized",
                "example_tickers",
            ],
            max_rows=40,
        ),
        "",
        "## Sector Rotation Context",
        "",
        "Point-in-time sector breadth, momentum acceleration, and flow acceleration. This context cannot authorize or rank a trade until prospective payoff validation passes.",
        "",
        _markdown_table(
            sector_rotation,
            [
                "sector",
                "sector_state",
                "sector_emergence_score",
                "sector_momentum_change_5s",
                "sector_breadth_change_5s",
                "sector_flow_acceleration",
                "sector_rotation_authority",
            ],
            max_rows=20,
        ),
        "",
        "## Strategy And Market Diagnostics",
        "",
        "Detailed diagnostics follow the Action Board above. Full gap-risk, OCO, flow/OI, target methodology, and audit fields remain in the swing-target CSV.",
    ]
    report_lines.extend(
        [
            "",
            "## Decision-Lane Audit",
            "",
            "Every `decision_eligible=True` row is shown here. A row is not an order unless its status is Execute; downstream payoff evidence and confirmation remain binding.",
            "",
            _markdown_table(compact_decision_candidates),
            "",
            "## Portfolio Repair / Open Risk",
            "",
            _markdown_table(compact_portfolio_repair, max_rows=20),
            "",
            f"_Full portfolio repair CSV: {paths['portfolio_repair']}_",
            "",
            "## $10k/month Target Math",
            "",
            _markdown_table(target_status),
            "",
            "### Detailed Math",
            "",
            "| Metric | Value |",
            "|:--|--:|",
            f"| Remaining monthly target | ${target_model['remaining_monthly_target']:,.2f} |",
            f"| Business days remaining | {target_model['business_days_remaining']} |",
            f"| Required daily P/L | ${target_model['required_daily_pl']:,.2f} |",
            f"| Required weekly P/L | ${target_model['required_weekly_pl']:,.2f} |",
            f"| Execute profit potential | ${target_model['execute_profit_potential']:,.2f} |",
            f"| Swing Target profit potential if filled | ${target_model['swing_target_profit_potential_if_filled']:,.2f} |",
            f"| Realistic fill-adjusted target potential | ${target_model['realistic_fill_adjusted_target_potential']:,.2f} |",
            f"| Aggregate risk budget applied | {'yes' if target_model['aggregate_risk_budget_applied'] else 'no'} |",
            f"| Visible / in-budget target tickets | {target_model['risk_bounded_target_ticket_count']} |",
            f"| Visible / in-budget max loss | ${target_model['risk_bounded_max_loss']:,.2f} |",
            f"| Visible / in-budget fill-adjusted potential | ${target_model['risk_bounded_fill_adjusted_target_potential']:,.2f} |",
            f"| Max risk if all target tickets fill | ${target_model['max_risk_if_all_target_tickets_fill']:,.2f} |",
            f"| Aggregate risk budget | {'not configured' if not target_model['aggregate_risk_budget_applied'] else '$' + format(target_model['risk_available'], ',.2f')} |",
            f"| Required average profit per trade | ${target_model['required_average_profit_per_trade']:,.2f} |",
            f"| Required number of target tickets | {target_model['required_number_of_target_tickets']} |",
            f"| Expected monthly run-rate | ${target_model['expected_monthly_run_rate']:,.2f} |",
            f"| Can sizing close the gap? | {'yes' if target_model['sizing_can_close_gap'] else 'no'} |",
            f"| Bottleneck | {target_model['exact_blocker_to_10k_month']} |",
            "",
            "### Setup-Family Expectations",
            "",
            _markdown_table(pd.DataFrame.from_dict(target_model["expected_by_setup_family"], orient="index").reset_index().rename(columns={"index": "setup family"})),
            "",
            "## No-Miss Audit",
            "",
            _markdown_table(compact_no_miss, max_rows=30),
            "",
            f"_Full no-miss audit CSV: {paths['no_miss_audit']}_",
            "",
            "### Candidate Generation Audit",
            "",
            _markdown_table(generation_audit, ["audit_item", "audit_status", "evidence"]),
            "",
            "### Safety & Calibration Audit",
            "",
            _markdown_table(
                safety_calibration,
                [
                    "setup_family_key",
                    "outcome_count",
                    "loss_count",
                    "loss_rate",
                    "expected_value",
                    "strategy_slump_muted",
                    "mute_until",
                    "shadow_backtest_status",
                ],
                max_rows=30,
            ),
            "",
            "### Hard Risk Cap Audit",
            "",
            _markdown_table(
                risk_cap_audit,
                [
                    "ticker",
                    "final disposition before",
                    "final disposition after",
                    "two_percent_loss_cap",
                    "max_loss_before",
                    "max_loss_after",
                    "risk_cap_action",
                    "risk_capped",
                ],
                max_rows=30,
            ),
            "",
            "### Secondary Liquidity Sweep",
            "",
            _markdown_table(
                secondary_sweep,
                [
                    "triggered",
                    "candidate_count",
                    "ticker",
                    "sweep_rank",
                    "flow_velocity_scan",
                    "rolling_5m_premium",
                    "vol_to_open_interest_ratio",
                    "sector_sweep",
                    "spy_qqq_correlation",
                    "beta_noise_ignored",
                    "final_sweep_action",
                ],
                max_rows=30,
            ),
            "",
            "## Opportunity Board",
            "",
            _markdown_table(compact_board, max_rows=30),
            "",
            f"_Full opportunity board CSV: {paths['opportunity_board']}_",
            "",
            "## Detailed artifacts",
            "",
            "| Artifact | Path |",
            "|:--|:--|",
        ]
    )
    for name, path in artifacts.items():
        report_lines.append(f"| {name} | {path} |")
    report_lines.append("")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return manifest


def _fallback_earnings_from_scored(scored: pd.DataFrame, *, asof: dt.date) -> dict[str, dt.date]:
    fallback: dict[str, dt.date] = {}
    if scored is None or scored.empty or "ticker" not in scored.columns:
        return fallback
    for _, row in scored.iterrows():
        ticker = _clean(row.get("ticker")).upper()
        parsed = pd.to_datetime(row.get("next_earnings_dt"), errors="coerce")
        if not ticker or pd.isna(parsed):
            continue
        event_date = parsed.date()
        if event_date < asof:
            continue
        existing = fallback.get(ticker)
        if existing is None or event_date < existing:
            fallback[ticker] = event_date
    return fallback


def _maximum_candidate_expiry(scored: pd.DataFrame, *, asof: dt.date) -> dt.date:
    if scored is None or scored.empty or "expiry" not in scored.columns:
        return asof + dt.timedelta(days=90)
    parsed = pd.to_datetime(scored["expiry"], errors="coerce").dropna()
    if parsed.empty:
        return asof + dt.timedelta(days=90)
    return min(parsed.max().date(), asof + dt.timedelta(days=120))


def run_v4_daily(*, base_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    asof = infer_asof_date(base_dir)
    input_provenance = build_input_provenance(base_dir)
    run_mode = RUN_MODE_V4
    try:
        stock_screener = load_stock_screener(base_dir, point_in_time=True)
        hot_chains = load_hot_chains(base_dir, asof, point_in_time=True)
    except Exception as exc:
        raise SystemExit(f"{PIPELINE_NAME_V4} input load failed: {exc}") from exc

    # Realised vol drives the credit policy's IV/HV richness gate, which is the
    # strongest premium-selling signal on the 2026 panel. It has to be computed
    # from the dated-folder close history; the UW `volatility` field is
    # outlier-contaminated and destroys the signal. See codexuw/realized_vol.py.
    stock_screener = attach_realized_vol(stock_screener, base_dir.parent, asof)
    sector_rotation = build_live_sector_rotation(base_dir.parent, asof=asof)
    try:
        chain_oi = load_chain_oi(base_dir, asof, point_in_time=True)
    except Exception:
        chain_oi = None

    regime = detect_regime(stock_screener)
    latest_asof = latest_dated_folder(base_dir.parent)
    note = live_planning_validation_note(asof, latest_asof)
    if note:
        regime["validation_note"] = note

    liquidity_shift = build_liquidity_shift_signals(
        base_dir=base_dir,
        root=base_dir.parent,
        asof=asof,
        stock_screener=stock_screener,
        hot_chains=hot_chains,
        chain_oi=chain_oi,
        regime=regime,
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
        point_in_time=True,
    )
    top_flow = liquidity_shift.get("top_flow_universe") if isinstance(liquidity_shift.get("top_flow_universe"), pd.DataFrame) else pd.DataFrame()
    flow_velocity = liquidity_shift.get("flow_velocity") if isinstance(liquidity_shift.get("flow_velocity"), pd.DataFrame) else pd.DataFrame()
    correlation = liquidity_shift.get("correlation_anomalies") if isinstance(liquidity_shift.get("correlation_anomalies"), pd.DataFrame) else pd.DataFrame()
    regime_context = _build_v4_regime_context(
        stock_screener=stock_screener,
        base_regime=regime,
        liquidity_shift=liquidity_shift,
        asof=asof,
        run_mode=run_mode,
    )

    pool = select_ticker_pool(stock_screener, max_tickers=args.max_tickers)
    pool = expand_pool_with_top_flow(pool, stock_screener, liquidity_shift, max_top_flow=50)
    index_pool = select_index_fallback_pool(stock_screener)
    bot_tickers = pool["ticker"].tolist() if not pool.empty and "ticker" in pool.columns else []
    if not index_pool.empty and "ticker" in index_pool.columns:
        bot_tickers = sorted(set(bot_tickers + index_pool["ticker"].tolist()))
    bot_flow = aggregate_bot_flow(
        base_dir,
        bot_tickers,
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
        allow_missing=True,
        point_in_time=True,
    )
    bot_flow_source_status = str(bot_flow.attrs.get("source_status") or "unknown")
    dark_pool_flow = aggregate_dark_pool_flow(
        base_dir,
        bot_tickers,
        max_rows=args.bot_max_rows if args.bot_max_rows > 0 else None,
        allow_missing=True,
        point_in_time=True,
    )
    dark_pool_source_status = str(dark_pool_flow.attrs.get("source_status") or "unknown")

    candidates = generate_candidates(
        pool,
        hot_chains,
        bot_flow,
        asof=asof,
        max_candidates=args.max_candidates,
        dark_pool_flow=dark_pool_flow,
        dark_pool_weight=args.dark_pool_weight,
    )
    if not index_pool.empty:
        index_candidates = generate_candidates(
            index_pool,
            hot_chains,
            bot_flow,
            asof=asof,
            max_candidates=12,
            index_fallback=True,
            dark_pool_flow=dark_pool_flow,
            dark_pool_weight=args.dark_pool_weight,
        )
        if not index_candidates.empty:
            candidates = pd.concat([candidates, index_candidates], ignore_index=True) if not candidates.empty else index_candidates
    fallback_candidates = build_fallback_income_candidates(
        stock_screener=stock_screener,
        hot_chains=hot_chains,
        liquidity_shift=liquidity_shift,
        asof=asof,
        max_candidates=12,
    )
    if not fallback_candidates.empty:
        candidates = (
            pd.concat([candidates, fallback_candidates], ignore_index=True)
            if not candidates.empty
            else fallback_candidates
        ).drop_duplicates(
            subset=["ticker", "direction", "expiry", "short_strike_eod", "long_strike_eod"],
            keep="first",
        )

    candidates["bot_flow_source_status"] = bot_flow_source_status
    candidates["dark_pool_source_status"] = dark_pool_source_status
    scored = live_validate_and_score(
        candidates,
        asof=asof,
        out_dir=out_dir,
        regime=regime,
        require_live=not args.offline,
        schwab_snapshot_dir=Path(args.schwab_snapshot_dir).expanduser().resolve() if args.schwab_snapshot_dir else None,
    )
    scored = apply_oi_carryover(scored, chain_oi)
    scored = apply_schwab_price_context(
        scored,
        out_dir=out_dir,
        asof=asof,
        offline=bool(args.offline),
    )
    scored = apply_replay_edge_model(
        scored,
        base_dir.parent / "out",
        asof=asof,
        history_namespace=EDGE_HISTORY_NAMESPACE,
    )

    if args.skip_portfolio or args.offline:
        portfolio = unavailable_portfolio_context("skipped" if args.skip_portfolio else "offline")
    else:
        try:
            portfolio = fetch_portfolio_context(
                out_dir,
                portfolio_income_mode=args.portfolio_income_mode,
                covered_income_allowed_tickers=[
                    ticker.strip().upper()
                    for ticker in str(args.covered_income_allowed_tickers or "").split(",")
                    if ticker.strip()
                ],
            )
        except Exception as exc:
            portfolio = unavailable_portfolio_context(str(exc))
    scored = apply_portfolio_context(scored, portfolio)

    if args.skip_catalysts:
        catalysts = None
    else:
        catalyst_tickers = sorted(set(scored["ticker"].dropna().astype(str).str.upper())) if not scored.empty and "ticker" in scored.columns else []
        event_exempt_tickers = {
            _clean(row.get("ticker")).upper()
            for _, row in scored.iterrows()
            if _clean(row.get("ticker")) and is_etf_row(row)
        }
        live_web_window = abs((dt.date.today() - asof).days) <= 7
        catalysts = load_catalyst_context(
            base_dir,
            catalyst_tickers,
            asof=asof,
            fallback_earnings=_fallback_earnings_from_scored(scored, asof=asof),
            resolve_web=bool(not args.offline and live_web_window),
            web_through=_maximum_candidate_expiry(scored, asof=asof),
            event_exempt_tickers=event_exempt_tickers,
        )
    if catalysts is not None:
        scored = apply_catalyst_context(scored, catalysts)
    macro = build_macro_event_gates(base_dir=base_dir, asof=asof, stock_screener=stock_screener, regime=regime)
    scored = apply_final_quality_guards(scored)
    scored = apply_high_conviction_decision_marks(scored, asof=asof)

    recent_performance = (
        {"status": "unavailable", "reason": "skipped"}
        if args.skip_recent_performance
        else load_recent_performance(
            base_dir.parent / "out",
            asof=asof,
            history_namespace=EDGE_HISTORY_NAMESPACE,
        )
    )
    live_outcomes = load_live_outcome_performance(base_dir.parent / "out")
    scored = apply_confirmation_framework(scored, asof=asof, regime=regime, recent_performance=recent_performance)
    scored = apply_confidence_components(scored, live_outcomes=live_outcomes)
    loss_review = load_recent_loss_review(base_dir.parent / "out", asof=asof, lookback_days=args.loss_lookback_days)
    scored = apply_loss_review(scored, loss_review)
    scored = apply_liquidity_shift_context(scored, liquidity_shift, require_intraday_vwap=False)
    confirmation = build_confirmation_evidence(scored=scored, asof=asof, input_provenance=input_provenance)
    scored = apply_confirmation_evidence(scored, confirmation)
    scored = assign_trade_statuses(scored, index_income_mode=args.index_income_mode)
    scored = apply_fallback_income_status(scored)

    data_quality = build_data_quality_status(
        input_provenance=input_provenance,
        scored=scored,
        portfolio=portfolio,
        catalysts=catalysts,
        recent_performance=recent_performance,
        live_outcomes=live_outcomes,
        run_mode=run_mode,
    )
    scored = apply_data_quality_gate(scored, data_quality)

    if args.daily_loss_limit > 0 and portfolio and portfolio.get("status") == "ok":
        if float(portfolio.get("day_pnl") or 0.0) <= -abs(args.daily_loss_limit):
            scored = scored.copy()
            scored["hard_rejects"] = scored.get("hard_rejects", pd.Series("", index=scored.index)).map(
                lambda value: _append_token(value, "daily_loss_limit")
            )

    return write_v4_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        asof=asof,
        args=args,
        candidates=candidates,
        scored=scored,
        board=pd.DataFrame(),
        top_flow=top_flow,
        flow_velocity=flow_velocity,
        correlation=correlation,
        macro=macro,
        confirmation=confirmation,
        data_quality=data_quality,
        portfolio=portfolio,
        regime=regime,
        regime_context=regime_context,
        recent_performance=recent_performance,
        live_outcomes=live_outcomes,
        loss_review=loss_review,
        liquidity_summary=liquidity_shift.get("summary", {}),
        uw_source_status={
            "stock_screener": "loaded",
            "hot_chains": "loaded",
            "chain_oi_changes": "loaded" if chain_oi is not None and not chain_oi.empty else "missing_or_empty",
            "bot_eod_report": bot_flow_source_status,
            "dp_eod_report": dark_pool_source_status,
            "dark_pool_weight": float(args.dark_pool_weight),
        },
        run_mode=run_mode,
        sector_rotation=sector_rotation,
    )


def _first_json(path: Path, patterns: list[str]) -> dict[str, Any]:
    for pattern in patterns:
        for candidate in sorted(path.glob(pattern)):
            data = _read_json(candidate)
            if data:
                return data
    return {}


def run_validation_harness_v4(
    *,
    root: Path,
    out_dir: Path,
    asof: dt.date,
    latest_n: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_systematic_date_folders(root, as_of=asof, latest_n=latest_n)
    selected_dates = [str(infer_asof_date(path)) for path in selected]
    snapshot_replay = build_snapshot_replay_summary(asof=asof, selected_dates=selected_dates)
    snapshot_replay_path = out_dir / f"codexdaily_v4_snapshot_replay_summary_{asof}.csv"
    snapshot_replay.to_csv(snapshot_replay_path, index=False)
    rows: list[dict[str, Any]] = []
    for folder in selected:
        day = infer_asof_date(folder)
        run_out = out_dir / f"codexdaily_v4_{day}"
        has_comparable_v4_manifest = False
        if getattr(args, "run_live", False):
            result = run_v4_daily(base_dir=folder, out_dir=run_out, args=args)
            has_comparable_v4_manifest = True
        else:
            source = root / "out" / f"codexdaily_v4_{day}"
            existing_manifest = _read_json(source / f"codexdaily_v4_manifest_{day}.json")
            if existing_manifest:
                result = existing_manifest
                has_comparable_v4_manifest = True
            else:
                # Do not rebuild a historical folder with today's Schwab quotes.
                # The stored exact-outcome replay below is the deterministic
                # validation source when a dated live-planning manifest is absent.
                result = {
                    "out_dir": str(run_out),
                    "opportunity_counts": {},
                    "swing_target_ticket_count": 0,
                    "historical_replay_status": "NO_STORED_V4_MANIFEST_NO_LIVE_REPRICE",
                }
        v4_counts = result.get("opportunity_counts") or {}
        target_model = result.get("target_model") or {}
        no_miss = result.get("no_miss_audit") or {}
        v3 = _read_json(root / "out" / f"codexdaily_v3_{day}" / f"codexdaily_v3_manifest_{day}.json")
        v2 = _read_json(root / "out" / f"codexdaily_v2_{day}" / f"codexuw_manifest_{day}.json")
        v1 = _first_json(root / "out" / f"codexdaily_v1_{day}", ["*manifest*.json", "*.json"])
        v3_counts = v3.get("opportunity_counts") or {}
        v2_funnel = v2.get("funnel") or {}
        rows.append(
            {
                "date": str(day),
                "v4_out_dir": result.get("out_dir", str(run_out)),
                "v4_swing_target_count": result.get("swing_target_ticket_count", 0) if has_comparable_v4_manifest else math.nan,
                "v4_execute_count": v4_counts.get("execute", 0) if has_comparable_v4_manifest else math.nan,
                "v4_scout_count": v4_counts.get("scout", 0) if has_comparable_v4_manifest else math.nan,
                "v4_visible_missed_opportunities": no_miss.get("suppressed_rows", 0) if has_comparable_v4_manifest else math.nan,
                "v4_over_filtering_signal": (
                    "yes" if v4_counts.get("execute", 0) == 0 and result.get("swing_target_ticket_count", 0) else "no"
                ) if has_comparable_v4_manifest else "not available - no stored V4 manifest",
                "v4_lane_coverage": ",".join(sorted((result.get("lane_coverage") or {}).keys())) if has_comparable_v4_manifest else "",
                "v4_target_feasibility": target_model.get("target_feasibility", "") if has_comparable_v4_manifest else "",
                "v4_target_math_blocker": target_model.get("exact_blocker_to_10k_month", "") if has_comparable_v4_manifest else "",
                "v4_no_miss_audit_completeness": (
                    "complete" if no_miss.get("top_flow_tickers_audited", 0) >= 50 else "partial"
                ) if has_comparable_v4_manifest else "not available - no stored V4 manifest",
                "v4_report_usefulness": "target sheet plus no-miss audit",
                "v4_historical_replay_status": result.get("historical_replay_status", "STORED_V4_MANIFEST"),
                "v3_swing_target_count": v3.get("swing_target_ticket_count", 0),
                "v3_execute_count": v3_counts.get("execute", 0),
                "v3_scout_count": v3_counts.get("scout", 0),
                "v2_execute_count": v2.get("execute_rows", v2_funnel.get("final_trade_rows", 0)),
                "v1_available": bool(v1),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / f"codexdaily_v4_validation_summary_{asof}.csv"
    summary.to_csv(summary_path, index=False)
    capacity_out = out_dir / "portfolio_capacity"
    capacity = write_portfolio_capacity_outputs(
        replay_detail=DEFAULT_EDGE_HISTORY_PATH,
        out_dir=capacity_out,
        monthly_target=float(args.monthly_profit_target),
        account_value=float(getattr(args, "validation_account_value", 0.0) or 0.0),
        aggregate_risk_budget=float(getattr(args, "risk_budget", 0.0) or 0.0),
        risk_per_trade_pct=float(getattr(args, "validation_risk_per_trade_pct", 0.02) or 0.02),
        max_contracts=max(1, int(getattr(args, "max_contracts_per_trade", 20) or 20)),
        max_ticker_share=float(getattr(args, "validation_max_active_ticker_share", 0.20) or 0.20),
        max_sector_share=float(getattr(args, "validation_max_active_sector_share", 0.40) or 0.40),
        asof=asof,
    )
    manifest = {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V4,
        "version_lock": pipeline_version_record("v4"),
        "run_mode": "validation",
        "asof": str(asof),
        "latest_n": latest_n,
        "selection_method": "latest source-complete dated folders at or before asof",
        "selected_dates": selected_dates,
        "summary_csv": str(summary_path),
        "snapshot_replay_summary_csv": str(snapshot_replay_path),
        "historical_replay_method": "stored exact outcomes only; no current Schwab quotes or portfolio state",
        "comparisons": rows,
        "portfolio_capacity": capacity,
    }
    manifest_path = out_dir / f"codexdaily_v4_validation_manifest_{asof}.json"
    report_path = out_dir / f"codexdaily_v4_validation_report_{asof}.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                f"# {PIPELINE_NAME_V4} Validation Report - {asof}",
                "",
                "## First Screen",
                "",
                "| Item | Value |",
                "|:--|:--|",
                f"| Pipeline | {PIPELINE_NAME_V4} |",
                f"| Version | {PIPELINE_VERSION_V4} |",
                f"| Version lock | locked {manifest['version_lock']['locked_on']}; rollback chain retained through v4.0 |",
                "| Run mode | Systematic recent-date validation |",
                f"| Selected dates | {', '.join(manifest['selected_dates'])} |",
                "",
                "## V4 vs V3/V2/V1",
                "",
                _markdown_table(summary),
                "",
                "## Deterministic Historical Snapshot Replay",
                "",
                "Uses stored dated-snapshot exact outcomes only. It does not call Schwab or rebuild old dates with current quotes.",
                "",
                _markdown_table(snapshot_replay),
                "",
                "## Overlapping-Book Capacity And Cost Stress",
                "",
                f"- Accepted exact replay trades: {capacity.get('accepted_trades', 0)}",
                f"- Risk-sized trades: {capacity.get('risk_sized_trades', 0)}",
                f"- Capacity status: {(capacity.get('feasibility') or {}).get('status', 'unknown')}",
                f"- Monthly target supported: {'YES' if (capacity.get('feasibility') or {}).get('target_supported') else 'NO'}",
                f"- Detailed report: {(capacity.get('artifacts') or {}).get('report', '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def run_v4_overlay(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    asof = _parse_date(args.date)
    if asof is None:
        raise SystemExit("--date is required")
    overlay_file = Path(args.overlay_file).expanduser().resolve()
    overlay_date = _parse_date(args.overlay_date) or _infer_overlay_date_from_name(overlay_file) or asof
    prior = Path(args.prior_out_dir).expanduser().resolve() if args.prior_out_dir else root / "out" / f"codexdaily_v4_{asof}"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "overlay", overlay_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _normalize_v4_dataframe(_read_prior_v4_scored(prior, asof))
    prior_manifest = _read_json(prior / f"codexdaily_v4_manifest_{asof}.json")
    regime = prior_manifest.get("market_regime") or {
        "trend": "range",
        "flow": "unknown",
        "volatility": "unknown",
        "transition": False,
    }

    after = live_validate_and_score(
        before,
        asof=overlay_date,
        out_dir=out_dir,
        regime=regime,
        require_live=True,
    )
    repriced_row_count = int(len(after))
    after["overlay_live_pricing_refreshed"] = True
    after["overlay_evaluation_date"] = str(overlay_date)
    if "expiry" in after.columns:
        expiry = pd.to_datetime(after["expiry"], errors="coerce")
        after["dte"] = (expiry - pd.Timestamp(overlay_date)).dt.days

    chain_oi = _load_overlay_chain_oi_file(overlay_file, asof=overlay_date)
    after = apply_oi_carryover(after, chain_oi)
    after = apply_schwab_price_context(
        after,
        out_dir=out_dir,
        asof=overlay_date,
        offline=False,
    )
    after = apply_replay_edge_model(
        after,
        root / "out",
        asof=overlay_date,
        history_namespace=EDGE_HISTORY_NAMESPACE,
    )

    try:
        portfolio = fetch_portfolio_context(out_dir, portfolio_income_mode="trading-sleeve-only")
    except Exception as exc:
        portfolio = unavailable_portfolio_context(str(exc))
    after = apply_portfolio_context(after, portfolio)

    base_dir = root / str(asof)
    catalyst_tickers = sorted(set(after["ticker"].dropna().astype(str).str.upper())) if not after.empty else []
    event_exempt_tickers = {
        _clean(row.get("ticker")).upper()
        for _, row in after.iterrows()
        if _clean(row.get("ticker")) and is_etf_row(row)
    }
    catalysts = load_catalyst_context(
        base_dir,
        catalyst_tickers,
        asof=overlay_date,
        fallback_earnings=_fallback_earnings_from_scored(after, asof=overlay_date),
        resolve_web=abs((dt.date.today() - overlay_date).days) <= 7,
        web_through=_maximum_candidate_expiry(after, asof=overlay_date),
        event_exempt_tickers=event_exempt_tickers,
    )
    after = apply_catalyst_context(after, catalysts)
    after = apply_final_quality_guards(after)
    after = apply_high_conviction_decision_marks(after, asof=overlay_date)
    recent_performance = load_recent_performance(
        root / "out",
        asof=overlay_date,
        history_namespace=EDGE_HISTORY_NAMESPACE,
    )
    live_outcomes = load_live_outcome_performance(root / "out")
    after = apply_confirmation_framework(
        after,
        asof=overlay_date,
        regime=regime,
        recent_performance=recent_performance,
    )
    after = apply_confidence_components(after, live_outcomes=live_outcomes)
    loss_review = load_recent_loss_review(root / "out", asof=overlay_date)
    after = apply_loss_review(after, loss_review)
    after = assign_trade_statuses(after, index_income_mode="primary")
    after = apply_fallback_income_status(after)
    _, overlay_confidence_calibration = build_default_walk_forward_calibration(asof=overlay_date)
    after = apply_confidence_calibration(after, overlay_confidence_calibration)
    payoff_summary, payoff_groups, payoff_walk_forward = build_default_payoff_calibration(asof=overlay_date)
    after = apply_payoff_calibration(after, payoff_groups)
    symbol_credit_summary, symbol_credit_evidence = build_symbol_credit_calibration(
        root,
        overlay_date,
        assessor=assess_credit_spread,
        screener_loader=load_stock_screener,
        history_path=DEFAULT_EDGE_HISTORY_PATH,
        credit_policy_version=CREDIT_POLICY_VERSION,
    )
    after = apply_symbol_credit_calibration(after, symbol_credit_summary, assessor=assess_credit_spread)
    symbol_credit_paths = write_symbol_credit_calibration_outputs(
        out_dir=out_dir,
        asof=f"{asof}_{overlay_date}",
        summary=symbol_credit_summary,
        evidence=symbol_credit_evidence,
    )
    walk_forward_credit_summary, walk_forward_credit_evidence, walk_forward_credit_model = (
        build_walk_forward_credit_model(asof=overlay_date)
    )
    after = apply_walk_forward_credit_model(after, walk_forward_credit_summary, walk_forward_credit_model)
    walk_forward_credit_paths = write_walk_forward_credit_outputs(
        out_dir=out_dir,
        asof=f"{asof}_{overlay_date}",
        summary=walk_forward_credit_summary,
        evidence=walk_forward_credit_evidence,
        qualified=after,
    )
    payoff_paths = write_payoff_calibration_outputs(
        out_dir=out_dir,
        asof=f"{asof}_{overlay_date}",
        summary=payoff_summary,
        groups=payoff_groups,
        walk_forward=payoff_walk_forward,
    )
    execution_evidence_integrity = build_execution_evidence_integrity(after)
    after = apply_v4_professional_dispositions(after, asof=overlay_date)
    strategy_validation_path = root / "out" / "sector_strategy_validation_v3.csv"
    strategy_validation = (
        pd.read_csv(strategy_validation_path, low_memory=False)
        if strategy_validation_path.exists()
        else pd.DataFrame()
    )
    strategy_registry = build_strategy_registry(
        payoff_summary=payoff_summary,
        payoff_groups=payoff_groups,
        confidence_summary=overlay_confidence_calibration,
        strategy_validation=strategy_validation,
    )
    after = apply_strategy_registry_gate(after, strategy_registry)
    after = after.copy()
    after["_overlay_exact_key"] = after.apply(_overlay_candidate_key, axis=1)
    after["_overlay_status_rank"] = after.apply(
        lambda row: _recommendation_rank(row.get("trade_status") or row.get("v4_disposition")),
        axis=1,
    )
    overlay_sort = ["_overlay_status_rank"]
    for column in ["decision_score", "live_execution_confidence", "score"]:
        if column in after.columns:
            after[column] = pd.to_numeric(after[column], errors="coerce")
            overlay_sort.append(column)
    after = (
        after.sort_values(overlay_sort, ascending=[False] * len(overlay_sort), na_position="last")
        .drop_duplicates("_overlay_exact_key", keep="first")
        .drop(columns=["_overlay_exact_key", "_overlay_status_rank"])
        .reset_index(drop=True)
    )
    after = apply_v4_credit_sleeve_cap(after)
    after = apply_v4_prospective_book_concentration(after)
    execution_reachability = build_execution_reachability_audit(after)
    top_flow = _read_prior_v4_top_flow(prior, asof)
    board = build_v4_opportunity_board(after, top_flow=top_flow)
    tickets = build_v4_swing_target_tickets(
        scored=after,
        board=board,
        regime=regime,
        top_flow=top_flow,
    )
    tickets, risk_cap_audit = apply_v4_risk_cap(tickets, portfolio)
    tickets = _attach_schwab_gex_context(tickets, out_dir / "schwab_chains")
    changes = _compare_v4_overlay_changes(before, after)
    scored_path = out_dir / f"codexdaily_v4_overlay_scored_reference_{asof}_{overlay_date}.csv"
    board_path = out_dir / f"codexdaily_v4_overlay_opportunity_board_{asof}_{overlay_date}.csv"
    changes_path = out_dir / f"codexdaily_v4_overlay_changes_{asof}_{overlay_date}.csv"
    tickets_path = out_dir / f"codexdaily_v4_overlay_swing_target_tickets_{asof}_{overlay_date}.csv"
    risk_cap_path = out_dir / f"codexdaily_v4_overlay_risk_cap_audit_{asof}_{overlay_date}.csv"
    reachability_path = out_dir / f"codexdaily_v4_overlay_execution_reachability_{asof}_{overlay_date}.json"
    after.to_csv(scored_path, index=False)
    board.to_csv(board_path, index=False)
    changes.to_csv(changes_path, index=False)
    tickets.to_csv(tickets_path, index=False)
    risk_cap_audit.to_csv(risk_cap_path, index=False)
    reachability_path.write_text(
        json.dumps(execution_reachability, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    execute_tickets = tickets[tickets["final disposition"].eq("Execute")]
    work_limit_tickets = tickets[tickets["final disposition"].eq("Swing Target / Work Limit")]
    scout_tickets = tickets[tickets["final disposition"].eq("Scout")]
    high_confidence_ticket_count = int(
        tickets["confidence rating"].fillna("").astype(str).str.contains("HIGH", case=False).sum()
    )
    change_summary = (
        changes["recommendation_change"].astype(str).value_counts().sort_index().to_dict()
        if not changes.empty and "recommendation_change" in changes.columns
        else {}
    )
    oi_context_changes = (
        int(
            changes["previous_oi_support"].fillna("").astype(str).ne(
                changes["new_oi_support_or_conflict"].fillna("").astype(str)
            ).sum()
        )
        if not changes.empty
        and {"previous_oi_support", "new_oi_support_or_conflict"}.issubset(changes.columns)
        else 0
    )
    manifest = {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V4,
        "version_lock": pipeline_version_record("v4"),
        "run_mode": "overlay",
        "asof": str(asof),
        "overlay_date": str(overlay_date),
        "prior_out_dir": str(prior),
        "overlay_file": str(overlay_file),
        "evaluation_date": str(overlay_date),
        "live_pricing_refreshed": True,
        "portfolio_status": portfolio.get("status", "unknown"),
        "edge_history_namespace": EDGE_HISTORY_NAMESPACE,
        "execution_evidence_integrity": execution_evidence_integrity,
        "execution_reachability": execution_reachability,
        "payoff_calibration": payoff_summary,
        "symbol_credit_calibration": symbol_credit_summary,
        "walk_forward_credit_calibration": walk_forward_credit_summary,
        "recent_performance": recent_performance,
        "repriced_candidate_rows_before_exact_dedupe": repriced_row_count,
        "candidate_rows_after_exact_dedupe": int(len(after)),
        "changed_candidate_rows": int(len(changes)),
        "oi_context_changed_rows": oi_context_changes,
        "recommendation_change_counts": change_summary,
        "swing_target_ticket_count": int(len(tickets)),
        "high_confidence_ticket_count": high_confidence_ticket_count,
        "execute_rows": int(len(execute_tickets)),
        "work_limit_rows": int(len(work_limit_tickets)),
        "scout_rows": int(len(scout_tickets)),
        "artifacts": {
            "overlay_scored_reference": str(scored_path),
            "overlay_opportunity_board": str(board_path),
            "overlay_changes": str(changes_path),
            "overlay_swing_target_tickets": str(tickets_path),
            "overlay_risk_cap_audit": str(risk_cap_path),
            "overlay_execution_reachability": str(reachability_path),
            **payoff_paths,
            **symbol_credit_paths,
            **walk_forward_credit_paths,
        },
        "scoring_source": "direct_v4_overlay",
    }
    manifest_path = out_dir / f"codexdaily_v4_overlay_manifest_{asof}_{overlay_date}.json"
    report_path = out_dir / f"codexdaily_v4_overlay_report_{asof}_{overlay_date}.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    manifest["artifacts"]["overlay_manifest"] = str(manifest_path)
    manifest["artifacts"]["overlay_report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_lines = [
        f"# {PIPELINE_NAME_V4} Overlay Report - {asof} with {overlay_date}",
        "",
        "## Action Board",
        "",
        (
            "This is the updated trading screen after applying the newer chain-OI file and refreshing "
            "Schwab pricing, portfolio context, and catalysts. **No order was placed.** Requote every "
            "structure and recheck news, earnings, portfolio overlap, and OCO instructions before entry."
        ),
        "",
        f"### 🟣 Enter Now / Execute — {len(execute_tickets)}",
        "",
        _markdown_table(_readable_action_ticket_table(execute_tickets)),
        "",
        "#### Enter Now ticket details",
        *_readable_ticket_detail_lines(execute_tickets),
        "",
        f"### 🟢 Work Limit / Approved — {len(work_limit_tickets)}",
        "",
        "These rows cleared setup-identification gates. Price, quote, or same-session evidence may still block entry; follow each row's stated condition.",
        "",
        _markdown_table(_readable_action_ticket_table(work_limit_tickets)),
        "",
        "#### Work Limit ticket details",
        *_readable_ticket_detail_lines(work_limit_tickets),
        "",
        f"### 🟡 Review / Scout — {len(scout_tickets)}",
        "",
        "Review rows are not trades and have no production execution authority.",
        "",
        _markdown_table(_readable_action_ticket_table(scout_tickets)),
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V4} |",
        f"| Version | {PIPELINE_VERSION_V4} |",
        f"| Version lock | locked {pipeline_version_record('v4')['locked_on']}; rollback chain retained through v4.0 |",
        "| Run mode | Next-session chain-OI overlay with refreshed live validation |",
        f"| Original analysis date | {asof} |",
        f"| Overlay evaluation date | {overlay_date} |",
        f"| Final visible tickets | {len(tickets)} |",
        f"| Enter Now / Execute | {len(execute_tickets)} |",
        f"| Work Limit / Approved | {len(work_limit_tickets)} |",
        f"| Review / Scout | {len(scout_tickets)} |",
        f"| High-confidence tickets | {high_confidence_ticket_count}; High requires both validated payoff and High-ready confidence evidence |",
        f"| Changed candidate comparisons | {len(changes)} |",
        f"| Rows with changed OI context | {oi_context_changes} |",
        f"| Evidence-authorized rows | {execution_reachability.get('evidence_authorized_rows', 0)} |",
        f"| Approved rows independent of current fill | {execution_reachability.get('approved_target_rows', 0)} |",
        f"| Final Execute rows before ticket risk-cap pass | {execution_reachability.get('final_execute_rows', 0)} |",
        f"| Market-open state affects identification | {execution_reachability.get('market_session_affects_identification', False)} |",
        "",
        "## Overlay Change Summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [{"Change": key, "Candidate comparisons": value} for key, value in change_summary.items()]
            )
        ),
        "",
        "The detailed opportunity board and candidate-level before/after audit remain CSV artifacts. They are not repeated here as partial trade ideas.",
        "",
        "## Detailed Artifacts",
        "",
        "| Artifact | Path |",
        "|:--|:--|",
    ]
    for name, path in manifest["artifacts"].items():
        report_lines.append(f"| {name} | {path} |")
    report_lines.append("")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(getattr(args, "root", DEFAULT_ROOT)).expanduser().resolve()
    if args.command == "run":
        base_dir = _base_dir_from_args(args)
        asof = infer_asof_date(base_dir)
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "run")
        manifest = run_v4_daily(base_dir=base_dir, out_dir=out_dir, args=args)
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        print(f"Swing target tickets: {manifest.get('swing_target_ticket_count', 0)}")
        print(f"Execute: {(manifest.get('opportunity_counts') or {}).get('execute', 0)}")
        print(f"Scout: {(manifest.get('opportunity_counts') or {}).get('scout', 0)}")
        return
    if args.command == "validate":
        asof = _parse_date(args.as_of) or latest_dated_folder(root)
        if asof is None:
            raise SystemExit("No dated folders found for validation")
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(root, asof, "validation")
        manifest = run_validation_harness_v4(root=root, out_dir=out_dir, asof=asof, latest_n=args.latest_n, args=args)
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        return
    if args.command == "overlay":
        manifest = run_v4_overlay(args)
        print(f"Wrote: {manifest.get('report_path')}")
        print(f"Manifest: {manifest.get('manifest_path')}")
        return


if __name__ == "__main__":
    main()
