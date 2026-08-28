"""Independent, point-in-time replay for the Groko production generator."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ._vendor import data as uw_data
from uwos import exact_spread_backtester
from uwos.exact_spread_backtester import HistoricalOptionQuoteStore, LegQuote
from groko import core


# Bound to core so the accepted-pin schema and the emitted schema can never
# drift apart again. A v4/v5 desync silently disabled the entire credit lane.
SCHEMA_VERSION = core.PINNED_REPLAY_MANIFEST_SCHEMA_VERSION
ROUND_TRIP_COMMISSION = 2.60
# Keep the outcome horizon and every management level coupled to the live
# selector, so replay evidence describes the population live may actually trade.
FIXED_HORIZON_SESSIONS = core.PLANNED_TRADE_HOLDING_SESSIONS
DEBIT_FIXED_HORIZON_SESSIONS = core.DEBIT_PLANNED_TRADE_HOLDING_SESSIONS
CREDIT_TAKE_PROFIT_REMAINING = core.CREDIT_TAKE_PROFIT_REMAINING
CREDIT_HARD_STOP_ENABLED = core.CREDIT_HARD_STOP_ENABLED
CREDIT_STOP_MULTIPLIER = 2.0
DEBIT_TAKE_PROFIT_MULTIPLIER = core.DEBIT_TAKE_PROFIT_MULTIPLIER
DEBIT_HARD_STOP_ENABLED = core.DEBIT_VERTICAL_HARD_STOP_ENABLED
DEBIT_STOP_REMAINING = core.DEBIT_STOP_REMAINING
LONG_OPTION_FIXED_HORIZON_SESSIONS = core.LONG_OPTION_PLANNED_TRADE_HOLDING_SESSIONS
LONG_OPTION_TAKE_PROFIT_MULTIPLIER = core.LONG_OPTION_TAKE_PROFIT_MULTIPLIER
LONG_OPTION_HARD_STOP_ENABLED = core.LONG_OPTION_HARD_STOP_ENABLED
LONG_OPTION_STOP_REMAINING = core.DEBIT_STOP_REMAINING
SUPPORTED_ROUTES = {
    "bull_call_debit",
    "bear_put_debit",
    "bull_put_credit",
    "bear_call_credit",
    "long_call",
    "long_put",
}
REQUIRED_REPLAY_SOURCES = (
    "stock_screener",
    "hot_chains",
    "chain_oi",
    "dp_eod",
)
OPTIONAL_REPLAY_SOURCES = ("bot_eod",)
REQUIRED_SOURCE_GAP_PREFIX = "required point-in-time replay sources missing"
# The compatible daily files retain point-in-time candidate rows before
# selection or outcomes. v1.56 changes selector/runtime behavior, not candidate
# discovery; it re-runs selection and outcomes from those dated rows. The
# v1.47 source additionally needs its already-reviewed reprice correction.
ENTRY_CACHE_COMPATIBILITY_POLICY = (
    "candidate-generation-v1.47-to-v1.78-risk-normalized-bull-call"
)
COMPATIBLE_ENTRY_CACHE_FINGERPRINTS: dict[str, str] = {
    "4e2301538ff4481a1918fb3515c474203d6ae009f165a1ff58937b843abf93b2": (
        "options-agent-v1.78-risk-normalized-bull-call-20260822-083000-pre-calibration-handoff"
    ),
    "c6cf48bb33ec62970a112054b66cf96d58bc34f644151b766d740b8d682fd8f1": (
        "options-agent-v1.77-independent-long-options-20260821-081510-pre-tier-fix"
    ),
    "e0ae6210a603a50778e7f404790a3c84546a2e15a3477bb35d8d6b93575eda95": (
        "options-agent-v1.75-credit-risk-parity-20260813-072836"
    ),
    "ff221872f78de05217f12111a4ce574a06d271032f7b481ca4a15a18ca4d2d3b": (
        "options-agent-v1.68-five-source-market-context-20260806-081546"
    ),
    "4d4f756964e4b27d831c62c28d3068eb3ca16ba4624baba6f5054b515fc782bc": (
        "options-agent-v1.50-mixed-credit-live-width-parity-20260722-094108"
    ),
    "75d950a8f8cc0e6daaa90de14271dbb369f6a4b04bb4eac98f4861cd42ed12b9": (
        "options-agent-v1.47-bull-put-route-isolation-20260722-082236"
    ),
}
V176_MIGRATED_DETAIL_COLUMNS = {
    "management_holding_exit_date",
    "management_holding_sessions",
    "selector_risk_exit_date",
    "selector_risk_horizon_sessions",
    "earnings_within_selector_risk_horizon",
    "promotion_outcome_available_date",
    "promotion_outcome_eligible",
}
DETAIL_COLUMNS = (
    "replay_row_id",
    "asof",
    "exit_day",
    "ticker",
    "full_name",
    "sector",
    "issue_type",
    "strategy",
    "strategy_route",
    "strategy_kind",
    "entry_side",
    "expiry",
    "dte",
    "regime",
    "candidate_bias",
    "flow_bias_label",
    "core_universe_member",
    "underlying_quality_tier",
    "marketcap",
    "macro_tape_candidate",
    "macro_tape_direction",
    "price_move_pct",
    "price_tape_source",
    "stock_price_eod",
    "short_strike_eod",
    "long_strike_eod",
    "short_leg_eod",
    "long_leg_eod",
    "planned_entry_date",
    "entry_dte",
    "entry_width",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_price",
    "target_entry_limit",
    "entry_credit",
    "entry_debit",
    "entry_credit_pct_width",
    "entry_debit_pct_width",
    "entry_quote_width_pct",
    "next_session_bid",
    "next_session_ask",
    "next_session_mid",
    "next_session_quote_width_pct",
    "next_session_reprice_observed",
    "next_session_reprice_approved",
    "next_session_reprice_reason",
    "executed_entry_price",
    "executed_entry_credit",
    "executed_entry_debit",
    "executed_target_exit",
    "executed_stop_exit",
    "reward_risk",
    "breakeven",
    "target_exit",
    "stop_exit",
    "expected_move_ratio",
    "combined_flow_bias",
    "flow_total_premium",
    "iv_rank",
    "iv30d",
    "source_contract_oi",
    "source_contract_volume",
    "next_earnings_dt",
    "earnings_before_expiry",
    "earnings_within_holding_horizon",
    "planned_holding_exit_date",
    "planned_holding_sessions",
    "management_holding_exit_date",
    "management_holding_sessions",
    "selector_risk_exit_date",
    "selector_risk_horizon_sessions",
    "earnings_within_selector_risk_horizon",
    "macro_event_count_before_expiry",
    "macro_event_count_within_holding_horizon",
    "trade_quality_status",
    "hard_rejects",
    "quality_gate_reason",
    "decision_pass",
    "decision_score",
    "exact_fillable",
    "fill_reason",
    "exact_evaluated",
    "exact_reason",
    "exit_value",
    "exit_trigger",
    "holding_sessions",
    "pnl_1x",
    "return_on_risk",
    "promotion_outcome_available_date",
    "promotion_outcome_eligible",
    "selected_for_policy",
    "selector_partition",
    "selection_rank_for_day",
    "selection_outcome_independent",
    "candidate_source",
    "dated_quote_sources",
    "producer",
)


def _number(value: Any) -> Optional[float]:
    return core._as_float(value)


def _is_required_source_gap(exc: Exception) -> bool:
    return isinstance(exc, FileNotFoundError) and str(exc).startswith(REQUIRED_SOURCE_GAP_PREFIX)


def _cache_fingerprint(candidate_limit: Optional[int]) -> str:
    """Bind daily caches to the exact generator/replay source and run policy."""

    digest = hashlib.sha256()
    for source_path in (
        Path(core.__file__).resolve(),
        Path(exact_spread_backtester.__file__).resolve(),
        Path(uw_data.__file__).resolve(),
        Path(__file__).resolve(),
    ):
        digest.update(source_path.read_bytes())
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    digest.update(f"candidate_limit={int(candidate_limit or 0)}".encode("utf-8"))
    digest.update(f"fixed_horizon={FIXED_HORIZON_SESSIONS}".encode("utf-8"))
    digest.update(f"debit_horizon={DEBIT_FIXED_HORIZON_SESSIONS}".encode("utf-8"))
    digest.update(f"long_option_horizon={LONG_OPTION_FIXED_HORIZON_SESSIONS}".encode("utf-8"))
    digest.update(f"commission={ROUND_TRIP_COMMISSION}".encode("utf-8"))
    return digest.hexdigest()


def _compatible_entry_cache(
    cached: pd.DataFrame,
    audit: Mapping[str, Any],
    *,
    signal_day: dt.date,
    discovery_limit: Optional[int],
) -> bool:
    """Allow an audited candidate-cache migration with reviewed semantic transforms."""

    fingerprint = str(audit.get("cache_fingerprint") or "")
    if discovery_limit or fingerprint not in COMPATIBLE_ENTRY_CACHE_FINGERPRINTS:
        return False
    if str(audit.get("day") or "") != signal_day.isoformat():
        return False
    if str(audit.get("error") or "").strip():
        return False
    if str(audit.get("required_source_status") or "").lower() != "pass":
        return False
    source_paths = audit.get("required_source_paths") or {}
    if not isinstance(source_paths, Mapping) or any(
        not list(source_paths.get(label) or []) for label in REQUIRED_REPLAY_SOURCES
    ):
        return False
    required_columns = set(DETAIL_COLUMNS) - V176_MIGRATED_DETAIL_COLUMNS
    if not required_columns.issubset(cached.columns):
        return False
    if not cached.empty:
        if cached["selected_for_policy"].map(core._truthy).any():
            return False
        if cached["exact_evaluated"].map(core._truthy).any():
            return False
        producers = set(cached["producer"].dropna().astype(str).str.strip())
        if producers - {"groko.replay"}:
            return False
        if set(cached["asof"].dropna().astype(str)) - {signal_day.isoformat()}:
            return False
    return True


def _migrate_v176_route_management(
    out: pd.DataFrame,
    *,
    root: Path,
    signal_day: dt.date,
) -> pd.DataFrame:
    """Re-annotate v1.76 management fields and erase every cached outcome."""

    out = out.copy()
    for column in V176_MIGRATED_DETAIL_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out["days_to_earnings"] = out.get(
        "next_earnings_dt",
        pd.Series("", index=out.index, dtype=object),
    ).map(
        lambda value: (
            (_date(value) - signal_day).days if _date(value) is not None else ""
        )
    )
    event_calendar = dict(core.load_options_event_calendar(root))
    event_calendar["corporate_events"] = []
    out = core.annotate_contract_event_risk(
        out,
        as_of=signal_day,
        event_calendar=event_calendar,
    )
    out["exit_day"] = out["management_holding_exit_date"]
    out["promotion_outcome_available_date"] = out["management_holding_exit_date"]
    out["promotion_outcome_eligible"] = False
    for idx, row in out.iterrows():
        entry_type = str(row.get("entry_side") or "").upper()
        width = _number(row.get("entry_width")) or 0.0
        source_entry = _number(row.get("entry_price")) or 0.0
        if entry_type in {"CREDIT", "DEBIT"} and source_entry > 0 and width > 0:
            target, stop = _management_levels(
                entry_type,
                source_entry,
                width,
                route=str(row.get("strategy_route") or ""),
            )
            out.at[idx, "target_exit"] = target
            out.at[idx, "stop_exit"] = stop if stop is not None else math.nan
        executed_entry = _number(row.get("executed_entry_price")) or 0.0
        if entry_type in {"CREDIT", "DEBIT"} and executed_entry > 0 and width > 0:
            target, stop = _management_levels(
                entry_type,
                executed_entry,
                width,
                route=str(row.get("strategy_route") or ""),
            )
            out.at[idx, "executed_target_exit"] = target
            out.at[idx, "executed_stop_exit"] = (
                stop if stop is not None else math.nan
            )
    for column in (
        "exit_value",
        "holding_sessions",
        "pnl_1x",
        "return_on_risk",
    ):
        out[column] = math.nan
    out["exact_evaluated"] = False
    out["exact_reason"] = out.get(
        "next_session_reprice_approved",
        pd.Series(False, index=out.index),
    ).map(
        lambda value: (
            "fixed_horizon_exit_quote_pending"
            if core._truthy(value)
            else "entry_not_approved"
        )
    )
    out["exit_trigger"] = ""
    out["selected_for_policy"] = False
    out["selector_partition"] = ""
    out["selection_rank_for_day"] = ""
    return out


def _augment_v177_long_option_candidates(
    cached: pd.DataFrame,
    *,
    root: Path,
    signal_day: dt.date,
    quote_store: Optional[HistoricalOptionQuoteStore] = None,
) -> pd.DataFrame:
    """Reconstruct missing long-option rows from dated pre-outcome candidates."""

    if cached.empty:
        return cached
    seeds = cached.copy()
    bias = seeds.get(
        "candidate_bias",
        pd.Series("", index=seeds.index, dtype=object),
    ).fillna("").astype(str).str.lower()
    tier = seeds.get(
        "underlying_quality_tier",
        pd.Series("", index=seeds.index, dtype=object),
    ).fillna("").astype(str).str.lower()
    issue_type = seeds.get(
        "issue_type",
        pd.Series("", index=seeds.index, dtype=object),
    ).fillna("").astype(str).str.upper()
    seeds = seeds[
        bias.isin({"bullish", "bearish"})
        & tier.isin({"core", "liquid"})
        & issue_type.eq("COMMON STOCK")
    ].copy()
    if seeds.empty:
        return cached
    seeds["__candidate_bias"] = seeds["candidate_bias"].astype(str).str.lower()
    seeds["__decision_score"] = pd.to_numeric(
        seeds.get("decision_score", pd.Series(0.0, index=seeds.index)),
        errors="coerce",
    ).fillna(0.0)
    seeds = seeds.sort_values(
        ["ticker", "__candidate_bias", "__decision_score"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop_duplicates(["ticker", "__candidate_bias"], keep="first")

    existing = {
        (
            str(row.get("ticker") or "").upper(),
            str(row.get("strategy_route") or ""),
        )
        for _, row in cached.iterrows()
    }
    dated_chain = core._dated_option_chain_for_construction(
        root / signal_day.isoformat(),
        signal_day,
    )
    if dated_chain.empty:
        return cached

    constructed_rows: list[dict[str, Any]] = []
    for _, seed in seeds.iterrows():
        direction = "Long Call" if seed["__candidate_bias"] == "bullish" else "Long Put"
        route = "long_call" if direction == "Long Call" else "long_put"
        ticker = str(seed.get("ticker") or "").upper()
        if (ticker, route) in existing:
            continue
        candidate = {
            "ticker": ticker,
            "full_name": seed.get("full_name", ""),
            "sector": seed.get("sector", ""),
            "issue_type": seed.get("issue_type", ""),
            "marketcap": seed.get("marketcap", ""),
            "close": seed.get("stock_price_eod", ""),
            "bias": seed["__candidate_bias"],
            "flow_bias_label": seed.get("flow_bias_label", seed["__candidate_bias"]),
            "candidate_source": seed.get("candidate_source", "flow_oi"),
            "macro_tape_candidate": seed.get("macro_tape_candidate", False),
            "macro_tape_direction": seed.get("macro_tape_direction", ""),
            "price_move_pct": seed.get("price_move_pct", ""),
            "price_tape_source": seed.get("price_tape_source", ""),
            "score": seed.get("decision_score", 0.0),
            "signal_premium": seed.get("flow_total_premium", 0.0),
            "combined_flow_bias": seed.get("combined_flow_bias", ""),
            "iv_rank": seed.get("iv_rank", ""),
            "iv30d": seed.get("iv30d", ""),
            "next_earnings_dt": seed.get("next_earnings_dt", ""),
            "total_open_interest": seed.get("source_contract_oi", ""),
            "total_volume": seed.get("source_contract_volume", ""),
            "core_universe_member": seed.get("core_universe_member", False),
            "underlying_quality_tier": seed.get("underlying_quality_tier", ""),
            "quality_status": "qualified",
        }
        constructed = core.construct_long_option(
            candidate,
            dated_chain,
            direction=direction,
        )
        if not str(constructed.get("trade_plan") or "").strip():
            continue
        constructed.update(
            {
                "close": candidate["close"],
                "sector": candidate["sector"],
                "core_universe_member": candidate["core_universe_member"],
                # The source cache already carries the point-in-time equity
                # quality tier. Do not recompute it from one option leg's
                # volume, which is not the stock's average daily volume.
                "underlying_quality_tier": candidate[
                    "underlying_quality_tier"
                ],
                "iv_rank": candidate["iv_rank"],
                "iv30d": candidate["iv30d"],
                "next_earnings_dt": candidate["next_earnings_dt"],
            }
        )
        route_metadata = core._strategy_route(
            route,
            f"v1.77 dated-cache {route} reconstruction",
        )
        constructed_rows.append(
            core._attach_strategy_route_metadata(constructed, route_metadata)
        )
    if not constructed_rows:
        return cached

    constructed_frame = pd.DataFrame(constructed_rows)
    constructed_frame["days_to_earnings"] = constructed_frame.get(
        "next_earnings_dt",
        pd.Series("", index=constructed_frame.index),
    ).map(
        lambda value: (
            (_date(value) - signal_day).days if _date(value) is not None else ""
        )
    )
    event_calendar = dict(core.load_options_event_calendar(root))
    event_calendar["corporate_events"] = []
    constructed_frame = core.annotate_contract_event_risk(
        constructed_frame,
        as_of=signal_day,
        event_calendar=event_calendar,
    )
    symbols = {
        str(leg.get("occ_symbol") or "").upper()
        for _, row in constructed_frame.iterrows()
        for leg in core._directed_registry_legs(row)
        if str(leg.get("occ_symbol") or "").strip()
    }
    store = quote_store or HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    entry_day = core._add_regular_market_days(signal_day, 1)
    target_quotes = _quote_index(store, signal_day, symbols)
    next_session_quotes = _quote_index(store, entry_day, symbols)
    regime = str(cached.get("regime", pd.Series("unknown")).iloc[0] or "unknown")
    replay_rows = [
        _replay_row(
            row,
            signal_day=signal_day,
            entry_day=entry_day,
            exit_day=core._add_regular_market_days(
                signal_day,
                core._management_holding_sessions(row),
            ),
            regime=regime,
            target_quote_index=target_quotes,
            next_session_quote_index=next_session_quotes,
        )
        for _, row in constructed_frame.iterrows()
    ]
    additions = pd.DataFrame(replay_rows, columns=DETAIL_COLUMNS)
    return pd.concat([cached.reindex(columns=DETAIL_COLUMNS), additions], ignore_index=True)


def _finish_v177_cache_migration(
    out: pd.DataFrame,
    migration: str,
    *,
    root: Optional[Path],
    signal_day: Optional[dt.date],
    quote_store: Optional[HistoricalOptionQuoteStore],
) -> tuple[pd.DataFrame, str]:
    if root is None or signal_day is None:
        return out, migration
    augmented = _augment_v177_long_option_candidates(
        out,
        root=root,
        signal_day=signal_day,
        quote_store=quote_store,
    )
    suffix = "v1_77_independent_long_option_reconstruction"
    return augmented, f"{migration}_then_{suffix}" if migration else suffix


def _repair_v177_long_option_quality_tiers(out: pd.DataFrame) -> pd.DataFrame:
    """Restore long rows' dated equity tier from their source candidate rows."""

    if out.empty or "strategy_route" not in out.columns:
        return out
    repaired = out.copy()
    routes = repaired["strategy_route"].fillna("").astype(str)
    long_mask = routes.isin(core.LONG_OPTION_ROUTES)
    if not long_mask.any():
        return repaired

    reference = repaired.loc[~long_mask].copy()
    reference["__ticker"] = reference.get(
        "ticker",
        pd.Series("", index=reference.index, dtype=object),
    ).fillna("").astype(str).str.upper()
    reference["__bias"] = reference.get(
        "candidate_bias",
        pd.Series("", index=reference.index, dtype=object),
    ).fillna("").astype(str).str.lower()
    reference["__tier"] = reference.get(
        "underlying_quality_tier",
        pd.Series("", index=reference.index, dtype=object),
    ).fillna("").astype(str).str.lower()
    reference = reference[
        reference["__ticker"].ne("")
        & reference["__bias"].isin({"bullish", "bearish"})
        & reference["__tier"].isin({"core", "liquid", "speculative"})
    ]
    if reference.empty:
        return repaired
    tier_rank = {"core": 0, "liquid": 1, "speculative": 2}
    reference["__tier_rank"] = reference["__tier"].map(tier_rank)
    tier_map = (
        reference.sort_values("__tier_rank", kind="mergesort")
        .drop_duplicates(["__ticker", "__bias"], keep="first")
        .set_index(["__ticker", "__bias"])["__tier"]
        .to_dict()
    )
    for idx in repaired.index[long_mask]:
        ticker = str(repaired.at[idx, "ticker"] or "").upper()
        bias = str(repaired.at[idx, "candidate_bias"] or "").lower()
        source_tier = tier_map.get((ticker, bias))
        if source_tier:
            repaired.at[idx, "underlying_quality_tier"] = source_tier
    return repaired


def _migrate_compatible_candidate_cache(
    cached: pd.DataFrame,
    *,
    source_fingerprint: str,
    root: Optional[Path] = None,
    signal_day: Optional[dt.date] = None,
    quote_store: Optional[HistoricalOptionQuoteStore] = None,
) -> tuple[pd.DataFrame, str]:
    """Reuse dated candidate rows while recording any required semantic migration."""

    out = cached.copy()
    if source_fingerprint not in COMPATIBLE_ENTRY_CACHE_FINGERPRINTS:
        return out, ""
    if source_fingerprint == "4e2301538ff4481a1918fb3515c474203d6ae009f165a1ff58937b843abf93b2":
        return out, "v1_78_selector_calibration_handoff_only"
    if source_fingerprint == "c6cf48bb33ec62970a112054b66cf96d58bc34f644151b766d740b8d682fd8f1":
        out = _repair_v177_long_option_quality_tiers(out)
        if root is None or signal_day is None:
            return out, "v1_77_long_option_dated_equity_tier_repair"
        out = _migrate_v176_route_management(
            out,
            root=root,
            signal_day=signal_day,
        )
        return out, "v1_77_long_option_dated_equity_tier_repair_and_outcome_reset"
    if source_fingerprint == "e0ae6210a603a50778e7f404790a3c84546a2e15a3477bb35d8d6b93575eda95":
        if root is None or signal_day is None:
            return out, "v1_75_route_management_and_maturity_reset"
        # v1.75 candidate discovery and exact entry repricing are unchanged.
        # Re-annotate event/management fields, then discard every outcome so
        # v1.76 can score the new route-specific exit policy from raw quotes.
        out = _migrate_v176_route_management(
            out,
            root=root,
            signal_day=signal_day,
        )
        return _finish_v177_cache_migration(
            out,
            "v1_75_route_management_and_maturity_reset",
            root=root,
            signal_day=signal_day,
            quote_store=quote_store,
        )
    if source_fingerprint == "ff221872f78de05217f12111a4ce574a06d271032f7b481ca4a15a18ca4d2d3b":
        # v1.68 candidate discovery and exact quotes remain valid. v1.75 only
        # tightens fresh credit-spread approval, so migrate the entry decision
        # from recorded point-in-time fields and recompute outcomes downstream.
        entry_type = out.get(
            "entry_side",
            pd.Series("", index=out.index, dtype=object),
        ).fillna("").astype(str).str.upper()
        entry_credit = pd.to_numeric(
            out.get("executed_entry_credit", pd.Series(math.nan, index=out.index)),
            errors="coerce",
        )
        width = pd.to_numeric(
            out.get("entry_width", pd.Series(math.nan, index=out.index)),
            errors="coerce",
        )
        previously_approved = out.get(
            "next_session_reprice_approved",
            pd.Series(False, index=out.index, dtype=bool),
        ).map(core._truthy)
        economics_reject = (
            previously_approved
            & entry_type.eq("CREDIT")
            & entry_credit.div(width).lt(core.MIN_SEND_NOW_CREDIT_WIDTH_RATIO)
        )
        if economics_reject.any():
            reason = (
                "next_session_reprice_quality_fail:send_now_credit_width_below_"
                f"{int(core.MIN_SEND_NOW_CREDIT_WIDTH_RATIO * 100)}pct"
            )
            out.loc[economics_reject, "next_session_reprice_approved"] = False
            out.loc[economics_reject, "next_session_reprice_reason"] = reason
            out.loc[economics_reject, "exact_reason"] = reason
            for column in (
                "executed_entry_price",
                "executed_entry_credit",
                "executed_entry_debit",
                "executed_target_exit",
                "executed_stop_exit",
                "exit_value",
                "holding_sessions",
                "pnl_1x",
                "return_on_risk",
            ):
                if column in out.columns:
                    out.loc[economics_reject, column] = math.nan
            if "exact_evaluated" in out.columns:
                out.loc[economics_reject, "exact_evaluated"] = False
        if root is None or signal_day is None:
            return out, "v1_68_send_now_credit_width_25pct"
        out = _migrate_v176_route_management(
            out,
            root=root,
            signal_day=signal_day,
        )
        return _finish_v177_cache_migration(
            out,
            "v1_68_credit_width_then_v1_76_route_management",
            root=root,
            signal_day=signal_day,
            quote_store=quote_store,
        )
    if source_fingerprint == "4d4f756964e4b27d831c62c28d3068eb3ca16ba4624baba6f5054b515fc782bc":
        # v1.50 already has the exact candidate and reprice semantics. v1.56
        # replays the current selector lane over the unselected dated rows.
        if root is None or signal_day is None:
            return out, "v1_50_selector_lane_only"
        out = _migrate_v176_route_management(
            out,
            root=root,
            signal_day=signal_day,
        )
        return _finish_v177_cache_migration(
            out,
            "v1_50_selector_lane_then_v1_76_route_management",
            root=root,
            signal_day=signal_day,
            quote_store=quote_store,
        )
    reason = out.get(
        "next_session_reprice_reason",
        pd.Series("", index=out.index, dtype=object),
    ).fillna("").astype(str)
    invalid_net_market = reason.eq("invalid_next_session_reprice_economics")
    if invalid_net_market.any():
        # v1.47 already found both exact legs; it only labeled an invalid net
        # market as unobserved. v1.48 correctly records that as known no-entry.
        out.loc[invalid_net_market, "next_session_reprice_observed"] = True
    quote_width = pd.to_numeric(
        out.get(
            "next_session_quote_width_pct",
            pd.Series(math.nan, index=out.index, dtype=float),
        ),
        errors="coerce",
    )
    previously_approved = out.get(
        "next_session_reprice_approved",
        pd.Series(False, index=out.index, dtype=bool),
    ).map(core._truthy)
    selector_width_reject = previously_approved & quote_width.gt(
        core.MAX_SELECTOR_ENTRY_QUOTE_WIDTH_PCT
    )
    if selector_width_reject.any():
        reason = (
            "next_session_reprice_quality_fail:selector_quote_width_above_"
            f"{core.MAX_SELECTOR_ENTRY_QUOTE_WIDTH_PCT:.2f}"
        )
        out.loc[selector_width_reject, "next_session_reprice_approved"] = False
        out.loc[selector_width_reject, "next_session_reprice_reason"] = reason
        out.loc[selector_width_reject, "exact_reason"] = reason
        for column in (
            "executed_entry_price",
            "executed_entry_credit",
            "executed_entry_debit",
            "executed_target_exit",
            "executed_stop_exit",
        ):
            if column in out.columns:
                out.loc[selector_width_reject, column] = ""
    if root is None or signal_day is None:
        return out, "v1_47_reprice_resolution_and_selector_quote_width_25pct"
    out = _migrate_v176_route_management(
        out,
        root=root,
        signal_day=signal_day,
    )
    return _finish_v177_cache_migration(
        out,
        "v1_47_reprice_then_v1_76_route_management",
        root=root,
        signal_day=signal_day,
        quote_store=quote_store,
    )


def _eligible_replay_days(
    available_dates: Sequence[dt.date],
    *,
    start: dt.date,
    end: dt.date,
) -> list[dt.date]:
    """Return regular-session signals with at least one observable exit session.

    Credit spreads are managed to a take-profit level and typically resolve far
    inside ``FIXED_HORIZON_SESSIONS``.  Requiring the full worst-case horizon to
    fit inside the data window discarded every signal in the most recent ~7
    weeks even when its outcome was fully determined, which left live runs
    trading on evidence that stopped almost two months earlier.  Admit any day
    that can be entered next session and observed at least once; outcomes that
    are still open at the data edge are censored during resolution rather than
    being counted as time exits.
    """

    return sorted(
        day
        for day in available_dates
        if (
            start <= day <= end
            and core.is_regular_market_day(day)
            and core._add_regular_market_days(day, 2) <= end
        )
    )


def _date(value: Any) -> Optional[dt.date]:
    return core._optional_iso_date(value)


def _spread_quotes(
    entry_type: str,
    short_quote: Optional[LegQuote],
    long_quote: LegQuote,
) -> tuple[float, float, float]:
    if entry_type == "CREDIT":
        bid = short_quote.bid - long_quote.ask
        ask = short_quote.ask - long_quote.bid
    else:
        bid = long_quote.bid - short_quote.ask
        ask = long_quote.ask - short_quote.bid
    mid = (bid + ask) / 2.0
    return float(bid), float(ask), float(mid)


def _entry_price(entry_type: str, bid: float, ask: float) -> float:
    return float(bid if entry_type == "CREDIT" else ask)


def _exit_value(entry_type: str, bid: float, ask: float) -> float:
    return float(ask if entry_type == "CREDIT" else bid)


def _next_session_reprice_status(
    row: Mapping[str, Any],
    *,
    entry_day: dt.date,
    regime: str,
    entry_type: str,
    target_limit: float,
    bid: float,
    ask: float,
    width: float,
    quote_width_pct: float,
    short_quote: LegQuote,
    long_quote: LegQuote,
) -> tuple[bool, bool, Optional[float], str]:
    """Mirror the production next-session exact-leg reprice gate.

    The EOD pipeline selects an exact structure from source-session fields, but
    it never promises the source natural quote as a durable next-session limit.
    Before entry, production refreshes the unchanged legs and accepts the order
    only when the newly quoted natural price still clears the original economic
    floor. Historical UW files expose daily closes rather than intraday quotes,
    so this is an end-of-session reprice proxy, not a claim that an intraday
    order would or would not have filled.
    """

    route = str(row.get("strategy_route") or "")
    long_option_route = route in core.LONG_OPTION_ROUTES
    invalid_market = (
        not all(
            math.isfinite(value)
            for value in (
                (target_limit, bid, ask, quote_width_pct)
                if long_option_route
                else (target_limit, bid, ask, width, quote_width_pct)
            )
        )
        or target_limit <= 0
        or bid < 0
        or ask <= 0
        or ask < bid
        or (not long_option_route and (width <= 0 or target_limit >= width))
    )
    if invalid_market:
        # Both exact legs were found, so an invalid net market is a known
        # no-entry outcome rather than missing quote evidence.
        return True, False, None, "invalid_next_session_reprice_economics"
    # Production emits a cent-precision net limit, so replay the same order
    # price instead of booking an untradeable floating-point net quote.
    entry = round(_entry_price(entry_type, bid, ask), 2)
    if entry <= 0 or (not long_option_route and entry >= width):
        return True, False, None, "invalid_next_session_reprice_economics"
    if entry_type == "CREDIT" and entry + 1e-9 < target_limit:
        return True, False, entry, "next_session_credit_below_source_target"
    if entry_type == "DEBIT" and entry - 1e-9 > target_limit:
        return True, False, entry, "next_session_debit_above_source_target"
    if entry_type not in {"CREDIT", "DEBIT"}:
        return False, False, None, "unsupported_entry_type"
    if (
        entry_type == "CREDIT"
        and entry / width + 1e-9 < core.MIN_SEND_NOW_CREDIT_WIDTH_RATIO
    ):
        return (
            True,
            False,
            entry,
            "next_session_reprice_quality_fail:send_now_credit_width_below_"
            f"{int(core.MIN_SEND_NOW_CREDIT_WIDTH_RATIO * 100)}pct",
        )
    if quote_width_pct > core.MAX_SELECTOR_ENTRY_QUOTE_WIDTH_PCT:
        return (
            True,
            False,
            entry,
            "next_session_reprice_quality_fail:selector_quote_width_above_"
            f"{core.MAX_SELECTOR_ENTRY_QUOTE_WIDTH_PCT:.2f}",
        )
    expiry = _date(row.get("expiry"))
    entry_dte = (expiry - entry_day).days if expiry is not None else -1
    regime = str(regime or row.get("regime") or "").lower()
    if entry_type == "CREDIT" and not 7 <= entry_dte <= 45:
        return True, False, entry, "next_session_credit_dte_outside_7_45"
    if entry_type == "CREDIT" and (
        route == "bear_call_credit"
        and regime == "risk_off"
        and entry_dte < core.MIN_RISK_OFF_BEAR_CALL_DTE
    ):
        return True, False, entry, "next_session_risk_off_bear_call_dte_below_minimum"
    if long_option_route:
        if entry_type != "DEBIT":
            return True, False, entry, "long_option_requires_debit_entry"
        if not core.MIN_LIVE_DTE <= entry_dte <= core.MAX_LIVE_DTE:
            return True, False, entry, "next_session_long_option_dte_outside_7_60"
    elif entry_type == "DEBIT":
        minimum_dte = 7 if route == "bull_call_debit" else 14
        if not minimum_dte <= entry_dte <= 45:
            return True, False, entry, "next_session_debit_dte_outside_route_policy"

    live_proxy = {
        "quote_width_pct": quote_width_pct,
        "short_oi": short_quote.open_interest if short_quote is not None else "",
        "short_volume": short_quote.volume if short_quote is not None else "",
        "long_oi": long_quote.open_interest,
        "long_volume": long_quote.volume,
        "dte": entry_dte,
    }
    if entry_type == "CREDIT":
        rejects = core._trade_quality_rejects(
            entry_credit=entry,
            credit_width_ratio=entry / width,
            max_loss=max((width - entry) * 100.0, 0.0),
            signal_premium=_number(row.get("signal_premium")) or 0.0,
            combined_flow_bias=_number(row.get("combined_flow_bias")) or 0.0,
            macro_tape_candidate=core._truthy(row.get("macro_tape_candidate")),
        )
    elif long_option_route:
        rejects = core._long_option_quality_rejects(
            entry_debit=entry,
            max_loss=entry * 100.0,
            signal_premium=_number(row.get("signal_premium")) or 0.0,
            combined_flow_bias=_number(row.get("combined_flow_bias")) or 0.0,
            macro_tape_candidate=core._truthy(row.get("macro_tape_candidate")),
        )
        rejects.extend(core._live_long_option_quality_rejects(live_proxy))
    else:
        rejects = core._debit_trade_quality_rejects(
            entry_debit=entry,
            debit_width_ratio=entry / width,
            max_profit=max((width - entry) * 100.0, 0.0),
            max_loss=entry * 100.0,
            signal_premium=_number(row.get("signal_premium")) or 0.0,
            combined_flow_bias=_number(row.get("combined_flow_bias")) or 0.0,
            macro_tape_candidate=core._truthy(row.get("macro_tape_candidate")),
        )
        rejects.extend(
            core._live_debit_contract_quality_rejects(
                live_proxy,
                dte=max(int(_number(row.get("dte")) or 0) - 1, 0),
            )
        )
    if not long_option_route:
        rejects.extend(core._live_spread_quality_rejects(live_proxy))
    rejects = core._dedupe_notes(rejects)
    if rejects:
        return True, False, entry, "next_session_reprice_quality_fail:" + ";".join(rejects)
    return True, True, entry, "next_session_reprice_approved"


def _exit_observation_dates(
    entry_day: dt.date,
    horizon_day: dt.date,
) -> list[dt.date]:
    """Return daily close observations strictly after a next-session entry."""

    dates: list[dt.date] = []
    current = entry_day
    while current < horizon_day:
        current = core._add_regular_market_days(current, 1)
        if current > horizon_day:
            break
        dates.append(current)
    return dates


def _bounded_exit_market(bid: float, ask: float, width: float) -> Optional[tuple[float, float]]:
    """Apply vertical-spread no-arbitrage bounds to an otherwise valid exit market."""

    if not all(math.isfinite(value) for value in (bid, ask, width)) or width <= 0 or ask < bid:
        return None
    return min(max(bid, 0.0), width), min(max(ask, 0.0), width)


def _management_levels(
    entry_type: str,
    entry: float,
    width: float,
    *,
    route: str = "",
) -> tuple[float, Optional[float]]:
    if entry_type == "CREDIT":
        stop = (
            round(min(width, entry * CREDIT_STOP_MULTIPLIER), 6)
            if CREDIT_HARD_STOP_ENABLED
            else None
        )
        return round(entry * CREDIT_TAKE_PROFIT_REMAINING, 6), stop
    if route in core.LONG_OPTION_ROUTES:
        return (
            round(entry * LONG_OPTION_TAKE_PROFIT_MULTIPLIER, 6),
            (
                round(max(entry * LONG_OPTION_STOP_REMAINING, 0.01), 6)
                if LONG_OPTION_HARD_STOP_ENABLED
                else None
            ),
        )
    return (
        round(min(width * 0.80, entry * DEBIT_TAKE_PROFIT_MULTIPLIER), 6),
        (
            round(max(entry * DEBIT_STOP_REMAINING, 0.01), 6)
            if DEBIT_HARD_STOP_ENABLED
            else None
        ),
    )


def _management_trigger(
    entry_type: str,
    value: float,
    target_exit: float,
    stop_exit: Optional[float],
    *,
    final_session: bool,
) -> str:
    if entry_type == "CREDIT":
        if value <= target_exit:
            return "take_profit"
        if stop_exit is not None and value >= stop_exit:
            return "stop_loss"
    else:
        if value >= target_exit:
            return "take_profit"
        if stop_exit is not None and value <= stop_exit:
            return "stop_loss"
    return "time_exit" if final_session else ""


def _pnl(entry_type: str, entry: float, exit_value: float) -> float:
    gross = entry - exit_value if entry_type == "CREDIT" else exit_value - entry
    return round(gross * 100.0 - ROUND_TRIP_COMMISSION, 2)


def _quote_index(
    quote_store: HistoricalOptionQuoteStore,
    day: dt.date,
    symbols: set[str],
) -> dict[str, LegQuote]:
    if not symbols:
        return {}
    quotes = quote_store._load_date_quotes(day, symbols)
    return quote_store._build_leg_quote_index(quotes)


def _candidate_legs(
    row: Mapping[str, Any],
) -> Optional[tuple[Optional[dict[str, Any]], dict[str, Any]]]:
    legs = core._directed_registry_legs(row)
    route = str(row.get("strategy_route") or "")
    if route in core.LONG_OPTION_ROUTES:
        if len(legs) != 1 or str(legs[0].get("side")).upper() != "BUY":
            return None
        return None, legs[0]
    if len(legs) != 2:
        return None
    short = next((leg for leg in legs if str(leg.get("side")).upper() == "SELL"), None)
    long = next((leg for leg in legs if str(leg.get("side")).upper() == "BUY"), None)
    if short is None or long is None:
        return None
    return short, long


def _expected_move_ratio(row: Mapping[str, Any], entry_type: str, entry: float) -> Optional[float]:
    spot = _number(row.get("close") or row.get("stock_price_eod"))
    dte = int(_number(row.get("dte")) or 0)
    iv = _number(row.get("iv30d"))
    if spot is None or spot <= 0 or dte <= 0 or iv is None or iv <= 0:
        return None
    if iv > 3.0:
        iv /= 100.0
    expected_move_pct = iv * math.sqrt(dte / 365.0)
    route = str(row.get("strategy_route") or "")
    short_strike = _number(row.get("short_strike"))
    long_strike = _number(row.get("long_strike"))
    if entry_type == "CREDIT" and short_strike is not None:
        breakeven = (
            short_strike + entry
            if route == "bear_call_credit"
            else short_strike - entry
        )
        distance = abs(breakeven - spot) / spot
        return distance / expected_move_pct if expected_move_pct > 0 else None
    if long_strike is None:
        return None
    breakeven = (
        long_strike + entry
        if route in {"bull_call_debit", "long_call"}
        else long_strike - entry
    )
    distance = (
        breakeven - spot
        if route in {"bull_call_debit", "long_call"}
        else spot - breakeven
    ) / spot
    return expected_move_pct / max(distance, 0.001)


def _dated_decision_pass(row: Mapping[str, Any]) -> bool:
    """Return the production pre-live quality decision using entry-time fields only."""

    recommendation_status = str(row.get("recommendation_status") or "").upper()
    quality_status = str(row.get("quality_status") or "").lower()
    trade_quality_status = str(row.get("trade_quality_status") or "").lower()
    return bool(
        quality_status == "qualified"
        and trade_quality_status != "rejected"
        and recommendation_status not in {"AVOID", "BLOCK"}
        and not str(row.get("hard_rejects") or "").strip()
    )


def _replay_row(
    row: Mapping[str, Any],
    *,
    signal_day: dt.date,
    entry_day: dt.date,
    exit_day: dt.date,
    regime: str,
    target_quote_index: Mapping[str, LegQuote],
    next_session_quote_index: Mapping[str, LegQuote],
) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    route = str(row.get("strategy_route") or "")
    long_option_route = route in core.LONG_OPTION_ROUTES
    entry_type = str(row.get("entry_type") or "").upper()
    expiry = _date(row.get("expiry"))
    legs = _candidate_legs(row)
    base_id = hashlib.sha256(
        f"{signal_day.isoformat()}|{ticker}|{row.get('trade_plan', '')}".encode("utf-8")
    ).hexdigest()[:24]
    planned_exit = min(expiry, exit_day) if expiry is not None else exit_day
    management_sessions = core._management_holding_sessions(row)
    result: dict[str, Any] = {
        "replay_row_id": base_id,
        "asof": signal_day.isoformat(),
        "exit_day": planned_exit.isoformat(),
        "ticker": ticker,
        "full_name": row.get("full_name", ""),
        "sector": row.get("sector", ""),
        "issue_type": row.get("issue_type", ""),
        "strategy": row.get("structure", ""),
        "strategy_route": route,
        "strategy_kind": entry_type,
        "entry_side": entry_type,
        "expiry": expiry.isoformat() if expiry else "",
        "dte": int(_number(row.get("dte")) or 0),
        "regime": regime,
        "candidate_bias": row.get("bias", ""),
        "flow_bias_label": row.get("flow_bias_label", ""),
        "core_universe_member": core._truthy(row.get("core_universe_member", False)),
        "underlying_quality_tier": row.get("underlying_quality_tier", ""),
        "marketcap": _number(row.get("marketcap")),
        "macro_tape_candidate": core._truthy(row.get("macro_tape_candidate", False)),
        "macro_tape_direction": row.get("macro_tape_direction", ""),
        "price_move_pct": _number(row.get("price_move_pct")),
        "price_tape_source": row.get("price_tape_source", ""),
        "stock_price_eod": _number(row.get("close")),
        "short_strike_eod": _number(row.get("short_strike")),
        "long_strike_eod": _number(row.get("long_strike")),
        "short_leg_eod": "",
        "long_leg_eod": "",
        "planned_entry_date": entry_day.isoformat(),
        "entry_dte": (expiry - entry_day).days if expiry is not None else "",
        "combined_flow_bias": _number(row.get("combined_flow_bias")),
        "flow_total_premium": _number(row.get("signal_premium")) or 0.0,
        "iv_rank": _number(row.get("iv_rank")),
        "iv30d": _number(row.get("iv30d")),
        "source_contract_oi": _number(row.get("total_open_interest")) or 0.0,
        "source_contract_volume": _number(row.get("total_volume")) or 0.0,
        "next_earnings_dt": str(row.get("next_earnings_dt") or "")[:10],
        "earnings_before_expiry": bool(row.get("earnings_before_expiry", False)),
        "earnings_within_holding_horizon": bool(
            row.get("earnings_within_holding_horizon", False)
        ),
        "planned_holding_exit_date": str(
            row.get("planned_holding_exit_date") or planned_exit.isoformat()
        )[:10],
        "planned_holding_sessions": int(
            _number(row.get("planned_holding_sessions"))
            or management_sessions
        ),
        "management_holding_exit_date": str(
            row.get("management_holding_exit_date") or planned_exit.isoformat()
        )[:10],
        "management_holding_sessions": int(
            _number(row.get("management_holding_sessions"))
            or management_sessions
        ),
        "selector_risk_exit_date": str(
            row.get("selector_risk_exit_date")
            or core._planned_trade_exit_date(
                signal_day,
                expiry,
                core.PLANNED_TRADE_HOLDING_SESSIONS,
            ).isoformat()
        )[:10],
        "selector_risk_horizon_sessions": int(
            _number(row.get("selector_risk_horizon_sessions"))
            or core.PLANNED_TRADE_HOLDING_SESSIONS
        ),
        "earnings_within_selector_risk_horizon": core._truthy(
            row.get("earnings_within_selector_risk_horizon", False)
        ),
        "macro_event_count_before_expiry": int(
            _number(row.get("macro_event_count_before_expiry")) or 0
        ),
        "macro_event_count_within_holding_horizon": int(
            _number(row.get("macro_event_count_within_holding_horizon")) or 0
        ),
        "trade_quality_status": row.get("trade_quality_status", ""),
        "hard_rejects": row.get("hard_rejects", ""),
        "quality_gate_reason": row.get("quality_gate_reason", ""),
        "decision_pass": _dated_decision_pass(row),
        "decision_score": _number(row.get("score")) or 0.0,
        "exact_fillable": False,
        "fill_reason": (
            "directed_one_leg_long_option_required"
            if long_option_route
            else "directed_two_leg_spread_required"
        ),
        "next_session_reprice_observed": False,
        "next_session_reprice_approved": False,
        "next_session_reprice_reason": "source_target_not_fillable",
        "exact_evaluated": False,
        "exact_reason": "entry_not_fillable",
        "promotion_outcome_available_date": planned_exit.isoformat(),
        "promotion_outcome_eligible": False,
        "selected_for_policy": False,
        "selector_partition": "",
        "selection_rank_for_day": "",
        "selection_outcome_independent": True,
        "candidate_source": row.get("candidate_source", ""),
        "dated_quote_sources": row.get("dated_quote_sources", ""),
        "producer": "groko.replay",
    }
    if legs is None or entry_type not in {"CREDIT", "DEBIT"} or expiry is None:
        return result
    short_leg, long_leg = legs
    short_symbol = (
        str(short_leg["occ_symbol"]).upper()
        if short_leg is not None
        else ""
    )
    long_symbol = str(long_leg["occ_symbol"]).upper()
    result["short_leg_eod"] = short_symbol
    result["long_leg_eod"] = long_symbol
    short_quote = target_quote_index.get(short_symbol) if short_symbol else None
    long_quote = target_quote_index.get(long_symbol)
    if long_quote is None or (not long_option_route and short_quote is None):
        result["fill_reason"] = "missing_exact_entry_leg_quote"
        return result
    if long_option_route:
        bid, ask, mid = float(long_quote.bid), float(long_quote.ask), float(long_quote.mid)
        width = 0.0
    else:
        assert short_quote is not None
        bid, ask, mid = _spread_quotes(entry_type, short_quote, long_quote)
        width = abs(
            (_number(row.get("short_strike")) or 0.0)
            - (_number(row.get("long_strike")) or 0.0)
        )
    entry = _entry_price(entry_type, bid, ask)
    quote_width = (ask - bid) / max(abs(mid), 0.01)
    default_target_limit = (
        round(entry, 2)
        if long_option_route
        else round(width * core.MIN_CREDIT_WIDTH_RATIO, 2)
        if entry_type == "CREDIT"
        else round(width * 0.45, 2)
    )
    target_limit = _number(row.get("target_entry")) or default_target_limit
    result.update(
        {
            "entry_width": round(width, 6),
            "entry_bid": round(bid, 6),
            "entry_ask": round(ask, 6),
            "entry_mid": round(mid, 6),
            "entry_price": round(entry, 6),
            "target_entry_limit": round(target_limit, 6),
            "entry_quote_width_pct": round(quote_width, 6),
            "source_contract_oi": (
                long_quote.open_interest
                if long_option_route
                else min(short_quote.open_interest, long_quote.open_interest)
            ),
            "source_contract_volume": (
                long_quote.volume
                if long_option_route
                else min(short_quote.volume, long_quote.volume)
            ),
        }
    )
    if (
        bid < 0
        or ask <= 0
        or ask < bid
        or entry <= 0
        or (not long_option_route and (width <= 0 or entry >= width))
    ):
        result["fill_reason"] = "invalid_conservative_entry_economics"
        return result
    if entry_type == "CREDIT":
        result["entry_credit"] = round(entry, 6)
        result["entry_credit_pct_width"] = round(entry / width, 6)
        max_profit = entry
        max_loss = width - entry
    elif long_option_route:
        result["entry_debit"] = round(entry, 6)
        result["entry_debit_pct_width"] = ""
        max_profit = entry * (LONG_OPTION_TAKE_PROFIT_MULTIPLIER - 1.0)
        max_loss = entry
    else:
        result["entry_debit"] = round(entry, 6)
        result["entry_debit_pct_width"] = round(entry / width, 6)
        max_profit = width - entry
        max_loss = entry
    result["reward_risk"] = round(max_profit / max_loss, 6) if max_loss > 0 else ""
    short_strike = _number(row.get("short_strike"))
    long_strike = _number(row.get("long_strike"))
    if entry_type == "CREDIT" and short_strike is not None:
        breakeven = (
            short_strike + entry
            if route == "bear_call_credit"
            else short_strike - entry
        )
    elif entry_type == "DEBIT" and long_strike is not None:
        breakeven = (
            long_strike + entry
            if route in {"bull_call_debit", "long_call"}
            else long_strike - entry
        )
    else:
        breakeven = _number(row.get("breakeven"))
    result["breakeven"] = breakeven
    target_exit, stop_exit = _management_levels(
        entry_type,
        entry,
        width,
        route=route,
    )
    result["target_exit"] = target_exit
    result["stop_exit"] = stop_exit if stop_exit is not None else ""
    result["expected_move_ratio"] = _expected_move_ratio(row, entry_type, entry)
    if entry_day >= planned_exit:
        result["fill_reason"] = "no_post_entry_exit_session"
        result["exact_reason"] = "no_post_entry_exit_session"
        return result
    result["exact_fillable"] = True
    result["fill_reason"] = "conservative_natural_target"
    next_short_quote = next_session_quote_index.get(short_symbol) if short_symbol else None
    next_long_quote = next_session_quote_index.get(long_symbol)
    if next_long_quote is None or (not long_option_route and next_short_quote is None):
        result["next_session_reprice_reason"] = "missing_next_session_entry_leg_quote"
        result["exact_reason"] = "missing_next_session_entry_leg_quote"
        return result
    if long_option_route:
        next_bid = float(next_long_quote.bid)
        next_ask = float(next_long_quote.ask)
        next_mid = float(next_long_quote.mid)
    else:
        assert next_short_quote is not None
        next_bid, next_ask, next_mid = _spread_quotes(
            entry_type,
            next_short_quote,
            next_long_quote,
        )
    next_quote_width = (next_ask - next_bid) / max(abs(next_mid), 0.01)
    reprice_observed, reprice_approved, executed_entry, reprice_reason = _next_session_reprice_status(
        row,
        entry_day=entry_day,
        regime=regime,
        entry_type=entry_type,
        target_limit=target_limit,
        bid=next_bid,
        ask=next_ask,
        width=width,
        quote_width_pct=next_quote_width,
        short_quote=next_short_quote,
        long_quote=next_long_quote,
    )
    result.update({
        "next_session_bid": round(next_bid, 6),
        "next_session_ask": round(next_ask, 6),
        "next_session_mid": round(next_mid, 6),
        "next_session_quote_width_pct": round(next_quote_width, 6),
        "next_session_reprice_observed": reprice_observed,
        "next_session_reprice_approved": reprice_approved,
        "next_session_reprice_reason": reprice_reason,
        "exact_reason": (
            "fixed_horizon_exit_quote_pending"
            if reprice_approved
            else reprice_reason
        ),
    })
    if reprice_approved and executed_entry is not None:
        executed_target_exit, executed_stop_exit = _management_levels(
            entry_type,
            executed_entry,
            width,
            route=route,
        )
        result["executed_entry_price"] = round(executed_entry, 6)
        result["executed_target_exit"] = round(executed_target_exit, 6)
        result["executed_stop_exit"] = (
            round(executed_stop_exit, 6) if executed_stop_exit is not None else ""
        )
        if entry_type == "CREDIT":
            result["executed_entry_credit"] = round(executed_entry, 6)
        else:
            result["executed_entry_debit"] = round(executed_entry, 6)
    return result


def _selector_frame(detail: pd.DataFrame) -> pd.DataFrame:
    frame = detail[detail["exact_fillable"].map(core._truthy)].copy()
    frame["signal_date"] = pd.to_datetime(frame["asof"], errors="coerce")
    frame["outcome_available_date"] = pd.to_datetime(
        frame["promotion_outcome_available_date"],
        errors="coerce",
    )
    frame["realized_pnl"] = pd.to_numeric(frame["pnl_1x"], errors="coerce")
    frame.loc[
        ~frame["promotion_outcome_eligible"].map(core._truthy),
        "realized_pnl",
    ] = math.nan
    frame["dte_bucket"] = frame["dte"].map(core._dte_bucket)
    frame["liquidity_bucket"] = frame.apply(core._liquidity_bucket, axis=1)
    frame["entry_type"] = frame["entry_side"]
    frame["economics_bucket"] = frame.apply(
        lambda row: core._economics_bucket(row, row.get("entry_type")),
        axis=1,
    )
    return frame


def _selected_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    realized_pnl = pd.to_numeric(
        selected.get("pnl_1x", pd.Series(dtype=float)),
        errors="coerce",
    )
    promotion_eligible = selected.get(
        "promotion_outcome_eligible",
        pd.Series(False, index=selected.index),
    ).map(core._truthy)
    realized_pnl = realized_pnl.where(promotion_eligible)
    executed = selected.get(
        "exact_evaluated",
        pd.Series(False, index=selected.index),
    ).map(core._truthy) & realized_pnl.notna()
    executed_rows = selected.loc[executed].copy()
    pnl = realized_pnl.loc[executed].dropna()
    reprice_observed_mask = (
        selected.get("next_session_reprice_observed", pd.Series(False, index=selected.index))
        .map(core._truthy)
    )
    reprice_approved_mask = (
        selected.get("next_session_reprice_approved", pd.Series(False, index=selected.index))
        .map(core._truthy)
    )
    reprice_observed = int(reprice_observed_mask.sum())
    reprice_approved = int(reprice_approved_mask.sum())
    known_no_entry = reprice_observed_mask & ~reprice_approved_mask
    execution_resolved = known_no_entry | executed
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf if gross_profit > 0 else 0.0
    return {
        "selected": int(len(selected)),
        "next_session_reprice_observed": int(reprice_observed),
        "next_session_reprice_observed_coverage": (
            round(float(reprice_observed) / float(len(selected)), 6) if len(selected) else 0.0
        ),
        "next_session_reprice_approved": int(reprice_approved),
        "next_session_reprice_approval_rate": (
            round(float(reprice_approved) / float(len(selected)), 6) if len(selected) else 0.0
        ),
        "evaluated": int(executed.sum()),
        "execution_resolution_coverage": (
            round(float(execution_resolved.sum()) / float(len(selected)), 6) if len(selected) else 0.0
        ),
        "outcome_coverage": round(float(executed.sum()) / float(len(selected)), 6) if len(selected) else 0.0,
        "selected_day_count": int(selected.get("asof", pd.Series(dtype=object)).astype(str).nunique()),
        "day_count": int(executed_rows.get("asof", pd.Series(dtype=object)).astype(str).nunique()),
        "selected_unique_tickers": int(selected.get("ticker", pd.Series(dtype=object)).astype(str).nunique()),
        "unique_tickers": int(executed_rows.get("ticker", pd.Series(dtype=object)).astype(str).nunique()),
        "avg_pnl_1x": round(float(pnl.mean()), 4) if not pnl.empty else None,
        "total_pnl_1x": round(float(pnl.sum()), 4) if not pnl.empty else None,
        "profit_factor": round(profit_factor, 6) if math.isfinite(profit_factor) else "inf",
        "win_rate": round(float((pnl > 0).mean()), 6) if not pnl.empty else None,
        "max_drawdown_1x": round(core._series_max_drawdown(pnl), 4) if not pnl.empty else None,
    }


def _candidate_rows_for_day(
    root: Path,
    signal_day: dt.date,
    quote_store: HistoricalOptionQuoteStore,
    *,
    discovery_limit: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    date_dir = root / signal_day.isoformat()
    inventory = core.build_source_inventory(date_dir, signal_day)
    inventory_sources = inventory.get("sources", {}) if isinstance(inventory, Mapping) else {}
    missing_sources = [
        label
        for label in REQUIRED_REPLAY_SOURCES
        if str((inventory_sources.get(label, {}) or {}).get("status") or "").lower() != "present"
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"required point-in-time replay sources missing for {signal_day.isoformat()}: "
            + ",".join(missing_sources)
        )
    source_paths = {
        label: list((inventory_sources.get(label, {}) or {}).get("paths") or [])
        for label in (*REQUIRED_REPLAY_SOURCES, *OPTIONAL_REPLAY_SOURCES)
    }
    missing_optional_sources = [
        label
        for label in OPTIONAL_REPLAY_SOURCES
        if str((inventory_sources.get(label, {}) or {}).get("status") or "").lower() != "present"
    ]
    source_audit = {
        "required_source_status": "pass",
        "required_source_paths": source_paths,
        "missing_optional_sources": missing_optional_sources,
    }
    raw, notes = core.build_raw_universe(
        date_dir,
        signal_day,
        discovery_limit=None,
    )
    market_price_regime = core.build_market_price_regime(raw, signal_day)
    raw = core.annotate_macro_tape_candidates(raw, market_price_regime)
    regime_payload = core.build_market_regime(
        raw,
        market_price_regime=market_price_regime,
    )
    regime = str(regime_payload.get("regime") or "unknown")
    candidates = core.generate_candidates(
        raw,
        limit=int(discovery_limit) if discovery_limit else None,
        market_price_regime=regime_payload,
    )
    priced, routing = core.price_candidates_with_routing_audit(
        date_dir,
        signal_day,
        candidates,
        root=None,
    )
    if priced.empty:
        return [], {
            "day": signal_day.isoformat(),
            "raw": len(raw),
            "candidates": len(candidates),
            "priced": 0,
            "rows": 0,
            "notes": notes,
            "market_price_regime": market_price_regime.get("regime", "unknown"),
            **source_audit,
        }
    merge_columns = [
        column
        for column in (
            "ticker",
            "close",
            "full_name",
            "sector",
            "marketcap",
            "core_universe_member",
            "total_open_interest",
            "total_volume",
            "next_earnings_dt",
        )
        if column in raw.columns
    ]
    if "ticker" in merge_columns:
        priced = priced.drop(
            columns=[column for column in merge_columns if column != "ticker" and column in priced.columns],
            errors="ignore",
        ).merge(raw[merge_columns].drop_duplicates("ticker"), on="ticker", how="left")
    priced = priced[
        priced.get("strategy_route", pd.Series("", index=priced.index)).isin(SUPPORTED_ROUTES)
        & priced.get("issue_type", pd.Series("", index=priced.index)).astype(str).str.upper().eq("COMMON STOCK")
        & priced.get("trade_plan", pd.Series("", index=priced.index)).astype(str).str.strip().ne("")
    ].copy()
    if priced.empty:
        return [], {
            "day": signal_day.isoformat(),
            "raw": len(raw),
            "candidates": len(candidates),
            "priced": 0,
            "rows": 0,
            "routing_rows": len(routing),
            "notes": notes,
            "market_price_regime": market_price_regime.get("regime", "unknown"),
            **source_audit,
        }
    event_calendar = dict(core.load_options_event_calendar(root))
    event_calendar["corporate_events"] = []
    priced["days_to_earnings"] = priced.get(
        "next_earnings_dt", pd.Series("", index=priced.index)
    ).map(
        lambda value: (
            (_date(value) - signal_day).days if _date(value) is not None else ""
        )
    )
    priced = core.annotate_contract_event_risk(
        priced,
        as_of=signal_day,
        event_calendar=event_calendar,
    )
    symbols = {
        str(leg["occ_symbol"]).upper()
        for _, row in priced.iterrows()
        for leg in (core._directed_registry_legs(row) or [])
    }
    target_quotes = _quote_index(quote_store, signal_day, symbols)
    entry_day = core._add_regular_market_days(signal_day, 1)
    next_session_quotes = _quote_index(quote_store, entry_day, symbols)
    rows = []
    for _, row in priced.iterrows():
        holding_sessions = core._management_holding_sessions(row)
        rows.append(
            _replay_row(
                row,
                signal_day=signal_day,
                entry_day=entry_day,
                exit_day=core._add_regular_market_days(
                    signal_day,
                    holding_sessions,
                ),
                regime=regime,
                target_quote_index=target_quotes,
                next_session_quote_index=next_session_quotes,
            )
        )
    return rows, {
        "day": signal_day.isoformat(),
        "raw": len(raw),
        "candidates": len(candidates),
        "priced": len(priced),
        "rows": len(rows),
        "target_math_valid": sum(bool(row["exact_fillable"]) for row in rows),
        "next_session_reprice_approved": sum(
            bool(row["next_session_reprice_approved"]) for row in rows
        ),
        "routing_rows": len(routing),
        "notes": notes,
        "market_price_regime": market_price_regime.get("regime", "unknown"),
        **source_audit,
    }


def run_independent_replay(
    root: Path,
    *,
    start: dt.date,
    end: dt.date,
    split_day: dt.date,
    output_dir: Path,
    discovery_limit: Optional[int] = None,
) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily_candidates"
    daily_dir.mkdir(parents=True, exist_ok=True)
    quote_store = HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    days = _eligible_replay_days(quote_store.available_dates(), start=start, end=end)
    cache_fingerprint = _cache_fingerprint(discovery_limit)
    all_rows: list[dict[str, Any]] = []
    day_audit: list[dict[str, Any]] = []
    compatible_entry_cache_days = 0
    compatible_entry_cache_source_fingerprints: set[str] = set()
    compatible_entry_cache_migrations: set[str] = set()
    for signal_day in days:
        cache_path = daily_dir / f"{signal_day.isoformat()}.csv"
        audit_path = daily_dir / f"{signal_day.isoformat()}.json"
        if cache_path.exists() and audit_path.exists():
            try:
                cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                cached_audit = {}
            try:
                cached = pd.read_csv(cache_path, low_memory=False)
            except (OSError, pd.errors.ParserError):
                cached = pd.DataFrame()
            exact_cache_match = (
                cached_audit.get("cache_fingerprint") == cache_fingerprint
                and set(DETAIL_COLUMNS).issubset(cached.columns)
            )
            compatible_cache_match = _compatible_entry_cache(
                cached,
                cached_audit,
                signal_day=signal_day,
                discovery_limit=discovery_limit,
            )
            if exact_cache_match or compatible_cache_match:
                audit_record = dict(cached_audit)
                audit_record["active_cache_fingerprint"] = cache_fingerprint
                if compatible_cache_match and not exact_cache_match:
                    source_fingerprint = str(cached_audit.get("cache_fingerprint") or "")
                    cached, migration = _migrate_compatible_candidate_cache(
                        cached,
                        source_fingerprint=source_fingerprint,
                        root=root,
                        signal_day=signal_day,
                        quote_store=quote_store,
                    )
                    audit_record["entry_cache_reused_from_fingerprint"] = source_fingerprint
                    audit_record["entry_cache_reused_from_pipeline_version"] = (
                        COMPATIBLE_ENTRY_CACHE_FINGERPRINTS[source_fingerprint]
                    )
                    audit_record["entry_cache_compatibility_policy"] = (
                        ENTRY_CACHE_COMPATIBILITY_POLICY
                    )
                    audit_record["entry_cache_compatibility_migration"] = migration
                    audit_record["rows"] = int(len(cached))
                    audit_record["v177_long_option_rows"] = int(
                        cached.get(
                            "strategy_route",
                            pd.Series("", index=cached.index),
                        ).isin(core.LONG_OPTION_ROUTES).sum()
                    )
                    compatible_entry_cache_days += 1
                    compatible_entry_cache_source_fingerprints.add(source_fingerprint)
                    if migration:
                        compatible_entry_cache_migrations.add(migration)
                all_rows.extend(cached.to_dict("records"))
                day_audit.append(audit_record)
                continue
        try:
            rows, audit = _candidate_rows_for_day(
                root,
                signal_day,
                quote_store,
                discovery_limit=discovery_limit,
            )
        except Exception as exc:
            rows = []
            if _is_required_source_gap(exc):
                audit = {
                    "day": signal_day.isoformat(),
                    "error": "",
                    "rows": 0,
                    "required_source_status": "excluded",
                    "source_gap_excluded": True,
                    "source_gap_reason": str(exc),
                }
            else:
                audit = {"day": signal_day.isoformat(), "error": str(exc), "rows": 0}
        audit["cache_fingerprint"] = cache_fingerprint
        pd.DataFrame(rows, columns=DETAIL_COLUMNS).to_csv(cache_path, index=False)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_rows.extend(rows)
        day_audit.append(audit)

    detail = pd.DataFrame(all_rows, columns=DETAIL_COLUMNS)
    if not detail.empty:
        promotion_dates = pd.to_datetime(
            detail["promotion_outcome_available_date"],
            errors="coerce",
        )
        detail["promotion_outcome_eligible"] = promotion_dates.le(pd.Timestamp(end))
        detail["exit_trigger"] = detail["exit_trigger"].fillna("").astype(str)
        due_symbols: dict[dt.date, set[str]] = {}
        fillable = detail["exact_fillable"].map(core._truthy) & detail[
            "next_session_reprice_approved"
        ].map(core._truthy)
        for _, row in detail[fillable].iterrows():
            entry_day = _date(row.get("planned_entry_date"))
            horizon_day = _date(row.get("exit_day"))
            if entry_day is None or horizon_day is None:
                continue
            symbols = {
                str(row.get("short_leg_eod") or "").upper(),
                str(row.get("long_leg_eod") or "").upper(),
            }
            for due in _exit_observation_dates(entry_day, horizon_day):
                if due <= end:
                    due_symbols.setdefault(due, set()).update(symbols)
        exit_quotes = {
            due: _quote_index(quote_store, due, {symbol for symbol in symbols if symbol})
            for due, symbols in due_symbols.items()
        }
        for idx, row in detail.iterrows():
            if not core._truthy(row.get("exact_fillable")):
                continue
            if not core._truthy(row.get("next_session_reprice_approved")):
                continue
            entry_day = _date(row.get("planned_entry_date"))
            horizon_day = _date(row.get("exit_day"))
            if entry_day is None:
                detail.at[idx, "exact_reason"] = "planned_entry_date_missing"
                continue
            if horizon_day is None:
                detail.at[idx, "exact_reason"] = "planned_exit_date_missing"
                continue
            horizon_fully_observable = horizon_day <= end
            exit_dates = _exit_observation_dates(entry_day, min(horizon_day, end))
            if not exit_dates:
                detail.at[idx, "exact_reason"] = (
                    "no_post_entry_exit_session"
                    if horizon_fully_observable
                    else "outcome_open_beyond_available_data"
                )
                continue
            entry_type = str(row.get("entry_side") or "").upper()
            route = str(row.get("strategy_route") or "")
            long_option_route = route in core.LONG_OPTION_ROUTES
            width = _number(row.get("entry_width")) or 0.0
            entry = _number(row.get("executed_entry_price")) or 0.0
            target_exit = _number(row.get("executed_target_exit"))
            stop_exit = _number(row.get("executed_stop_exit"))
            if target_exit is None:
                target_exit, stop_exit = _management_levels(
                    entry_type,
                    entry,
                    width,
                    route=str(row.get("strategy_route") or ""),
                )
            last_failure = "missing_exact_exit_leg_quote"
            for session_number, due in enumerate(exit_dates, start=1):
                quotes = exit_quotes.get(due, {})
                short_quote = quotes.get(str(row.get("short_leg_eod") or "").upper())
                long_quote = quotes.get(str(row.get("long_leg_eod") or "").upper())
                if long_quote is None or (not long_option_route and short_quote is None):
                    last_failure = "missing_exact_exit_leg_quote"
                    continue
                if long_option_route:
                    raw_bid, raw_ask = float(long_quote.bid), float(long_quote.ask)
                    if (
                        not all(math.isfinite(value) for value in (raw_bid, raw_ask))
                        or raw_bid < 0
                        or raw_ask < raw_bid
                    ):
                        last_failure = "invalid_crossed_exit_economics"
                        continue
                    bid, ask = raw_bid, raw_ask
                else:
                    assert short_quote is not None
                    raw_bid, raw_ask, _ = _spread_quotes(entry_type, short_quote, long_quote)
                    bounded_market = _bounded_exit_market(raw_bid, raw_ask, width)
                    if bounded_market is None:
                        last_failure = "invalid_crossed_exit_economics"
                        continue
                    bid, ask = bounded_market
                value = _exit_value(entry_type, bid, ask)
                trigger = _management_trigger(
                    entry_type,
                    value,
                    target_exit,
                    stop_exit,
                    final_session=(
                        horizon_fully_observable and due == exit_dates[-1]
                    ),
                )
                if not trigger:
                    continue
                pnl = _pnl(entry_type, entry, value)
                max_loss = (
                    (width - entry) * 100.0
                    if entry_type == "CREDIT"
                    else entry * 100.0
                )
                detail.at[idx, "exit_day"] = due.isoformat()
                detail.at[idx, "exit_value"] = round(value, 6)
                detail.at[idx, "exit_trigger"] = trigger
                detail.at[idx, "holding_sessions"] = session_number
                detail.at[idx, "pnl_1x"] = pnl
                detail.at[idx, "return_on_risk"] = round(pnl / max_loss, 6) if max_loss > 0 else ""
                detail.at[idx, "exact_evaluated"] = True
                bounded_suffix = (
                    "_no_arbitrage_bounded"
                    if not long_option_route and (bid != raw_bid or ask != raw_ask)
                    else ""
                )
                detail.at[idx, "exact_reason"] = f"conservative_{trigger}_liquidation{bounded_suffix}"
                break
            if not core._truthy(detail.at[idx, "exact_evaluated"]):
                detail.at[idx, "exact_reason"] = (
                    last_failure
                    if horizon_fully_observable
                    else "outcome_open_beyond_available_data"
                )

        policy = next(
            item
            for item in core.SELECTOR_CHALLENGER_POLICIES
            if item["policy_id"] == core.PROMOTED_SELECTOR_POLICY_ID
        )
        selected = core._select_challenger_policy_rows(_selector_frame(detail), policy)
        if not selected.empty:
            selected = selected.sort_values(
                ["signal_date", "__selector_economic_score", "ticker"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            selected["selection_rank_for_day"] = selected.groupby("signal_date").cumcount() + 1
            rank_by_id = dict(zip(selected["replay_row_id"], selected["selection_rank_for_day"]))
            detail["selected_for_policy"] = detail["replay_row_id"].isin(rank_by_id)
            detail["selection_rank_for_day"] = detail["replay_row_id"].map(rank_by_id).fillna("")
        detail["selector_partition"] = detail["asof"].map(
            lambda value: "pre_split" if _date(value) and _date(value) <= split_day else "heldout_test"
        )

    detail_path = output_dir / "groko_replay_detail.csv"
    detail.to_csv(detail_path, index=False)
    day_audit_path = output_dir / "groko_replay_day_audit.csv"
    pd.DataFrame(day_audit).to_csv(day_audit_path, index=False)
    selected_detail = detail[detail["selected_for_policy"].map(core._truthy)].copy() if not detail.empty else detail
    source_gap_rows = [row for row in day_audit if core._truthy(row.get("source_gap_excluded"))]
    failed_day_rows = [
        row
        for row in day_audit
        if not core._truthy(row.get("source_gap_excluded"))
        and str(row.get("error") or "").strip()
    ]
    included_day_rows = [
        row
        for row in day_audit
        if not core._truthy(row.get("source_gap_excluded"))
        and not str(row.get("error") or "").strip()
    ]
    successful_day_count = len(included_day_rows)
    optional_source_coverage = {
        label: {
            "present_days": sum(
                label not in set(row.get("missing_optional_sources") or [])
                for row in included_day_rows
            ),
            "missing_days": sum(
                label in set(row.get("missing_optional_sources") or [])
                for row in included_day_rows
            ),
        }
        for label in OPTIONAL_REPLAY_SOURCES
    }
    metrics = {
        "overall": _selected_metrics(selected_detail),
        "pre_split": _selected_metrics(
            selected_detail[selected_detail["selector_partition"].eq("pre_split")]
        ),
        "post_split_development": _selected_metrics(
            selected_detail[selected_detail["selector_partition"].eq("heldout_test")]
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": "groko.replay",
        "pipeline_version": core.PIPELINE_VERSION,
        "selector_policy_id": core.PROMOTED_SELECTOR_POLICY_ID,
        "selector_policy_development_cutoff": core.PROMOTED_SELECTOR_DEVELOPMENT_CUTOFF,
        "production_validation": False,
        "validation_scope": "development_replay_not_prospective_proof",
        "selection_outcome_independent": True,
        "selection_basis": "Options Agent dated candidate generation and entry-time fields only",
        "entry_price_policy": (
            "source-session plan selects fixed legs and a minimum economic floor; the exact legs are "
            "repriced at the next-session conservative natural quote before an outcome is scored"
        ),
        "next_session_reprice_policy": (
            "credit repricing requires the next-session natural bid at or above the source floor; "
            "debit repricing requires the next-session natural ask at or below the source floor; "
            "both must pass the available quote and liquidity checks"
        ),
        "exit_policy": (
            "conservative exact-leg daily target checks after the next-session entry; credit "
            f"targets buy-back at {CREDIT_TAKE_PROFIT_REMAINING:g}x the entry credit and carries "
            + (
                f"a hard stop at {CREDIT_STOP_MULTIPLIER:g}x the entry credit"
                if CREDIT_HARD_STOP_ENABLED
                else "no hard stop (the spread width is the defined risk)"
            )
            + f"; credit mandatory exit is signal session {FIXED_HORIZON_SESSIONS} or expiry; "
            f"debit verticals target {DEBIT_TAKE_PROFIT_MULTIPLIER:g}x entry debit, carry "
            + (
                f"a hard stop at {DEBIT_STOP_REMAINING:g}x remaining value"
                if DEBIT_HARD_STOP_ENABLED
                else "no hard stop (the spread debit is the defined risk)"
            )
            + f", and exit on signal session {DEBIT_FIXED_HORIZON_SESSIONS} or expiry; "
            f"long options target {LONG_OPTION_TAKE_PROFIT_MULTIPLIER:g}x entry debit, "
            + (
                f"stop at {LONG_OPTION_STOP_REMAINING:g}x remaining value"
                if LONG_OPTION_HARD_STOP_ENABLED
                else "carry no hard stop"
            )
            + f", and exit on signal session {LONG_OPTION_FIXED_HORIZON_SESSIONS} or expiry"
        ),
        "promotion_outcome_policy": (
            "realized P/L becomes selector evidence only after the route's complete planned "
            "holding window is observable; early take-profits from immature cohorts remain "
            "excluded so open losers cannot be censored"
        ),
        "credit_holding_sessions": FIXED_HORIZON_SESSIONS,
        "debit_holding_sessions": DEBIT_FIXED_HORIZON_SESSIONS,
        "credit_take_profit_remaining": CREDIT_TAKE_PROFIT_REMAINING,
        "credit_hard_stop_enabled": CREDIT_HARD_STOP_ENABLED,
        "credit_stop_multiplier": CREDIT_STOP_MULTIPLIER if CREDIT_HARD_STOP_ENABLED else None,
        "debit_take_profit_multiplier": DEBIT_TAKE_PROFIT_MULTIPLIER,
        "debit_hard_stop_enabled": DEBIT_HARD_STOP_ENABLED,
        "debit_stop_remaining": DEBIT_STOP_REMAINING if DEBIT_HARD_STOP_ENABLED else None,
        "long_option_holding_sessions": LONG_OPTION_FIXED_HORIZON_SESSIONS,
        "long_option_take_profit_multiplier": LONG_OPTION_TAKE_PROFIT_MULTIPLIER,
        "long_option_hard_stop_enabled": LONG_OPTION_HARD_STOP_ENABLED,
        "long_option_stop_remaining": (
            LONG_OPTION_STOP_REMAINING if LONG_OPTION_HARD_STOP_ENABLED else None
        ),
        "round_trip_commission": ROUND_TRIP_COMMISSION,
        "point_in_time_export_ceiling": True,
        "point_in_time_export_policy": "each candidate uses only its dated UW folder",
        "production_discovery_parity": not bool(discovery_limit),
        "candidate_limit": int(discovery_limit or 0),
        "cache_fingerprint": cache_fingerprint,
        "compatible_entry_cache_days": compatible_entry_cache_days,
        "compatible_entry_cache_source_fingerprints": sorted(
            compatible_entry_cache_source_fingerprints
        ),
        "compatible_entry_cache_migrations": sorted(compatible_entry_cache_migrations),
        "entry_cache_compatibility_policy": (
            ENTRY_CACHE_COMPATIBILITY_POLICY if compatible_entry_cache_days else "none"
        ),
        "max_days": 0,
        "days": successful_day_count,
        "observed_days": len(day_audit),
        "successful_days": successful_day_count,
        "failed_days": len(failed_day_rows),
        "source_coverage_status": "pass" if not failed_day_rows else "block",
        "excluded_required_source_gap_days": len(source_gap_rows),
        "excluded_required_source_gap_details": [
            {"day": row.get("day", ""), "reason": row.get("source_gap_reason", "")}
            for row in source_gap_rows
        ],
        "required_source_labels": list(REQUIRED_REPLAY_SOURCES),
        "optional_source_labels": list(OPTIONAL_REPLAY_SOURCES),
        "optional_source_coverage": optional_source_coverage,
        "failed_day_details": [
            {"day": row.get("day", ""), "error": row.get("error", "")}
            for row in failed_day_rows
        ],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "observation_end": end.isoformat(),
        "signal_end": max(
            (str(row.get("day") or "") for row in included_day_rows),
            default="",
        ),
        "required_outcome_horizon_sessions": FIXED_HORIZON_SESSIONS,
        "split_day": split_day.isoformat(),
        "detail_rows": int(len(detail)),
        "target_math_valid_rows": int(
            detail.get("exact_fillable", pd.Series(dtype=bool)).map(core._truthy).sum()
        ),
        "fillable_rows": int(
            detail.get("exact_fillable", pd.Series(dtype=bool)).map(core._truthy).sum()
        ),
        "next_session_reprice_observed_rows": int(
            detail.get("next_session_reprice_observed", pd.Series(dtype=bool)).map(core._truthy).sum()
        ),
        "next_session_reprice_approved_rows": int(
            detail.get("next_session_reprice_approved", pd.Series(dtype=bool)).map(core._truthy).sum()
        ),
        "evaluated_rows": int(detail.get("exact_evaluated", pd.Series(dtype=bool)).map(core._truthy).sum()),
        "selected_rows": int(len(selected_detail)),
        "metrics": metrics,
        "source_root": str(root),
        "detail_sha256": hashlib.sha256(detail_path.read_bytes()).hexdigest(),
        "day_audit_sha256": hashlib.sha256(day_audit_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / "groko_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "detail": detail_path,
        "manifest": manifest_path,
        "day_audit": day_audit_path,
    }


def write_replay_pin(root: Path, paths: Mapping[str, Path], *, split_day: dt.date) -> Path:
    root = Path(root).expanduser().resolve()
    detail_path = Path(paths["detail"]).resolve()
    manifest_path = Path(paths["manifest"]).resolve()
    day_audit_path = Path(paths["day_audit"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        str(manifest.get("source_coverage_status") or "").lower() != "pass"
        or int(manifest.get("failed_days") or 0) != 0
        or int(manifest.get("successful_days") or 0) != int(manifest.get("days") or 0)
    ):
        raise ValueError("cannot pin an Options Agent replay with incomplete source-day coverage")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("producer") != "groko.replay"
        or manifest.get("pipeline_version") != core.PIPELINE_VERSION
        or not bool(manifest.get("point_in_time_export_ceiling"))
        or not bool(manifest.get("selection_outcome_independent"))
        or not bool(manifest.get("production_discovery_parity"))
        or set(manifest.get("required_source_labels") or []) != set(REQUIRED_REPLAY_SOURCES)
        or set(manifest.get("optional_source_labels") or []) != set(OPTIONAL_REPLAY_SOURCES)
        or int(manifest.get("candidate_limit") or 0) != 0
        or int(manifest.get("max_days") or 0) != 0
        or manifest.get("day_audit_sha256") != hashlib.sha256(day_audit_path.read_bytes()).hexdigest()
    ):
        raise ValueError("cannot pin a capped, stale, or non-point-in-time Options Agent replay")
    if str(manifest.get("cache_fingerprint") or "").strip() != _cache_fingerprint(None):
        raise ValueError(
            "cannot pin an Options Agent replay generated by a different candidate-generation fingerprint"
        )
    optional_coverage = manifest.get("optional_source_coverage") or {}
    manifest_days = int(manifest.get("days") or 0)
    if any(
        int((optional_coverage.get(label) or {}).get("present_days") or 0)
        + int((optional_coverage.get(label) or {}).get("missing_days") or 0)
        != manifest_days
        for label in OPTIONAL_REPLAY_SOURCES
    ):
        raise ValueError("cannot pin replay with incomplete optional-source coverage disclosure")
    compatible_days = int(manifest.get("compatible_entry_cache_days") or 0)
    compatible_fingerprints = set(
        manifest.get("compatible_entry_cache_source_fingerprints") or []
    )
    if compatible_days > 0 and (
        manifest.get("entry_cache_compatibility_policy") != ENTRY_CACHE_COMPATIBILITY_POLICY
        or not compatible_fingerprints
        or not compatible_fingerprints.issubset(COMPATIBLE_ENTRY_CACHE_FINGERPRINTS)
    ):
        raise ValueError("cannot pin replay with unaudited compatible entry caches")
    if compatible_days == 0 and (
        compatible_fingerprints
        or manifest.get("entry_cache_compatibility_policy") not in {None, "", "none"}
    ):
        raise ValueError("cannot pin replay with inconsistent entry-cache provenance")
    day_audit = pd.read_csv(day_audit_path, low_memory=False)
    reused = day_audit.get(
        "entry_cache_reused_from_fingerprint",
        pd.Series(dtype=object),
    ).dropna().astype(str).str.strip()
    reused = reused[reused.ne("")]
    if len(reused) != compatible_days or set(reused) != compatible_fingerprints:
        raise ValueError("cannot pin replay with mismatched entry-cache audit counts")
    out_root = root / "out"
    payload = {
        "schema_version": "groko.replay_pin.v2",
        "replay_detail_path": str(detail_path.relative_to(out_root)),
        "replay_detail_sha256": hashlib.sha256(detail_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path.relative_to(out_root)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "day_audit_path": str(day_audit_path.relative_to(out_root)),
        "day_audit_sha256": hashlib.sha256(day_audit_path.read_bytes()).hexdigest(),
        "expected_history_days": int(manifest["days"]),
        "split_day": split_day.isoformat(),
        "producer": "groko.replay",
        "pipeline_version": manifest["pipeline_version"],
        "cache_fingerprint": manifest["cache_fingerprint"],
        "candidate_limit": 0,
        "production_discovery_parity": True,
        "production_validation": False,
    }
    pin_path = root / "knowledge" / "groko_replay_pin.json"
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    pin_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pin_path


def _parse_day(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start", type=_parse_day, required=True)
    parser.add_argument("--end", type=_parse_day, required=True)
    parser.add_argument("--split-day", type=_parse_day, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--discovery-limit",
        type=int,
        default=0,
        help="Optional diagnostic candidate cap; zero preserves production full-universe discovery.",
    )
    parser.add_argument("--pin", action="store_true")
    args = parser.parse_args(argv)
    paths = run_independent_replay(
        args.root,
        start=args.start,
        end=args.end,
        split_day=args.split_day,
        output_dir=args.output_dir,
        discovery_limit=args.discovery_limit or None,
    )
    if args.pin:
        paths = {**paths, "pin": write_replay_pin(args.root, paths, split_day=args.split_day)}
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
