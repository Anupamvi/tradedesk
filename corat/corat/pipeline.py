"""CORAT end-to-end research orchestration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.clock import today_new_york
from corat.config import PROJECT_ROOT, UniverseItem, discover_universe, load_universe, supporting_tickers
from corat.constants import DATA_UNAVAILABLE, RESEARCH_ONLY, SCHEMA_VERSION, TARGET_TRADE
from corat.context import event_risks, load_context, ticker_context
from corat.earnings import fetch_forward_earnings_calendar
from corat.history import analyze_analogues
from corat.models import HistoricalStats, OptionStructure, SourceTrace, object_dict
from corat.options import choose_option_structure, evaluate_option_evidence
from corat.orats import FetchBundle, OratsClient
from corat.regime import classify_market, rank_sectors
from corat.report import render_board_csv, render_report
from corat.scoring import build_stock_plan, choose_vehicle, model_stock_economics, score_candidate
from corat.schwab import SchwabClient, SchwabError, merge_quote_bar, quote_is_fresh, quote_to_bar
from corat.setups import detect_setups
from corat.store import canonical_json, read_json, sha256_bytes, sha256_file, utc_now, write_json, write_text
from corat.technical import append_core_spot, bars_from_dailies, technical_snapshot
from corat.volatility import normalize_volatility


def _paths(config: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    def resolve(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else PROJECT_ROOT / path
    return resolve(config["output_root"]), resolve(config["cache_root"]), resolve(config["state_root"])


def _index_latest(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        existing = result.get(ticker)
        if existing is None or str(row.get("updatedAt") or row.get("tradeDate") or "") >= str(existing.get("updatedAt") or existing.get("tradeDate") or ""):
            result[ticker] = row
    return result


def _empty_history(primary_horizon: int) -> HistoricalStats:
    return HistoricalStats(
        method=DATA_UNAVAILABLE,
        sample_size=0,
        reliable=False,
        horizon_returns={},
        primary_horizon=primary_horizon,
        win_rate=None,
        expectancy=None,
        average_winner=None,
        average_loser=None,
        profit_factor=None,
        mae=None,
        mfe=None,
        max_drawdown=None,
        signal_dates=[],
        primary_returns=[],
        primary_paths=[],
    )


def _empty_option(reason: str) -> OptionStructure:
    return OptionStructure(
        valid=False,
        strategy="NOT ENRICHED",
        expiration="",
        dte=0,
        legs=[],
        debit_credit="",
        expected_entry=None,
        natural_entry=None,
        maximum_loss=None,
        maximum_gain=None,
        breakeven=None,
        reward_risk=None,
        delta=None,
        gamma=None,
        theta=None,
        vega=None,
        theta_holding_cost=None,
        orats_theoretical_value=None,
        theoretical_edge=None,
        implied_volatility=None,
        quote_trade_date="",
        quote_updated_at="",
        reasons=[reason],
    )


def _positioning(chain_rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any], volatility: Mapping[str, Any]) -> Dict[str, Any]:
    call_levels = sorted(chain_rows, key=lambda row: int(float(row.get("callOpenInterest") or 0)), reverse=True)[:3]
    put_levels = sorted(chain_rows, key=lambda row: int(float(row.get("putOpenInterest") or 0)), reverse=True)[:3]
    call_text = ", ".join("{} {} OI {}".format(str(row.get("expirDate") or "")[:10], row.get("strike"), int(float(row.get("callOpenInterest") or 0))) for row in call_levels if float(row.get("callOpenInterest") or 0) > 0)
    put_text = ", ".join("{} {} OI {}".format(str(row.get("expirDate") or "")[:10], row.get("strike"), int(float(row.get("putOpenInterest") or 0))) for row in put_levels if float(row.get("putOpenInterest") or 0) > 0)
    flows = context.get("options_flow") or []
    if flows:
        flow_summary = "; ".join(str(row.get("claim") or row.get("title") or "") for row in flows)
    else:
        calls = int(volatility.get("call_volume") or 0)
        puts = int(volatility.get("put_volume") or 0)
        flow_summary = "ORATS aggregate call/put volume {}/{}; direction and opening/closing intent are unknown.".format(calls, puts)
    return {
        "major_call_oi_levels": call_text or DATA_UNAVAILABLE,
        "major_put_oi_levels": put_text or DATA_UNAVAILABLE,
        "gamma_context": "DATA UNAVAILABLE — CORAT does not infer dealer long/short positioning from OI.",
        "flow_summary": flow_summary,
    }


def _primary_risk(candidate: Mapping[str, Any]) -> str:
    risks = list(candidate.get("hard_rejections") or []) + list(candidate.get("blockers") or [])
    if candidate.get("earnings_crossed"):
        risks.insert(0, "Earnings falls inside the intended holding window")
    return risks[0] if risks else "Thesis failure at the defined technical invalidation; gaps can exceed the planned stop."


def _apply_correlation_caps(candidates: List[Dict[str, Any]], maximum: int) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for candidate in candidates:
        if candidate.get("status") != TARGET_TRADE:
            continue
        key = (str(candidate.get("sector") or "Unknown"), str((candidate.get("setup") or {}).get("direction") or "NEUTRAL"))
        current = counts.get(key, 0)
        if current >= maximum:
            candidate["portfolio_conflict"] = True
            candidate.setdefault("review_notes", []).append(
                "Correlated exposure warning: this is target {} for the same {} {} bucket; choose or resize during portfolio review.".format(current + 1, key[0], key[1])
            )
        counts[key] = current + 1


def _dedupe_traces(traces: Iterable[SourceTrace]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for trace in traces:
        key = (trace.endpoint, trace.cache_path, trace.status)
        if key in seen:
            continue
        seen.add(key)
        result.append(trace.to_dict())
    return result


def _enrichment_plan(
    prepared_order: Sequence[str],
    context_tickers: Iterable[str],
    enrichment_limit: int,
    option_limit: int,
    triggered_tickers: Iterable[str] = (),
) -> Tuple[set, List[str]]:
    context_set = {str(name).upper() for name in context_tickers}
    triggered_set = {str(name).upper() for name in triggered_tickers}
    context_names = [name for name in prepared_order if name in context_set]
    triggered_names = [name for name in prepared_order if name in triggered_set]
    top_names = list(prepared_order[: max(0, enrichment_limit)])
    enriched_names = set(top_names) | set(context_names) | set(triggered_names)
    if triggered_names:
        # Every currently triggered underlying is evaluated. A preliminary
        # technical score or untriggered context name may never consume a chain
        # slot ahead of an executable setup.
        return enriched_names, triggered_names
    option_priority = context_names + [name for name in top_names if name not in context_names]
    return enriched_names, option_priority[: max(0, option_limit)]


def _economics_rank(candidate: Mapping[str, Any]) -> Tuple[float, ...]:
    economics = candidate.get("economics") or {}
    vehicle = str(candidate.get("vehicle") or "")
    if vehicle == "OPTIONS":
        expected_return = economics.get("expected_return_on_max_loss")
        lower_return = economics.get("expected_return_lower_95_on_max_loss")
        expected_profit = economics.get("expected_profit_dollars")
        robust_profit = economics.get("robust_expected_profit_dollars")
        robust_positive = 1.0 if economics.get("robust_positive_across_fill_iv_stress") else 0.0
    else:
        expected_return = economics.get("expected_return_on_capital")
        basis = float((candidate.get("stock_plan") or {}).get("risk_basis_price") or (candidate.get("technical") or {}).get("price") or 0.0)
        lower_profit = economics.get("expected_profit_lower_95_per_share")
        lower_return = float(lower_profit) / basis if lower_profit is not None and basis > 0 else None
        expected_profit = economics.get("expected_profit_per_share")
        robust_profit = lower_profit
        robust_positive = 1.0 if lower_profit is not None and float(lower_profit) > 0 else 0.0
    return (
        robust_positive,
        float(lower_return) if lower_return is not None else -float("inf"),
        float(expected_return) if expected_return is not None else -float("inf"),
        float(economics.get("modeled_pop")) if economics.get("modeled_pop") is not None else -float("inf"),
        float(robust_profit) if robust_profit is not None else -float("inf"),
        float(expected_profit) if expected_profit is not None else -float("inf"),
        float(economics.get("model_sample_size") or 0),
        float(candidate.get("score") or 0),
    )


def _ranking_reason(candidate: Mapping[str, Any]) -> str:
    economics = candidate.get("economics") or {}
    if candidate.get("vehicle") == "OPTIONS":
        return (
            "Ranked by held-out option economics: POP {}, expected profit ${}, return on max loss {}, 95% lower bound ${}, "
            "natural/IV stress floor ${}; contextual score {} is only the final tie-breaker."
        ).format(
            economics.get("modeled_pop"),
            economics.get("expected_profit_dollars"),
            economics.get("expected_return_on_max_loss"),
            economics.get("expected_profit_lower_95_dollars"),
            economics.get("robust_expected_profit_dollars"),
            candidate.get("score"),
        )
    return (
        "Ranked by stock path economics: POP {}, expected profit/share ${}, return on capital {}, 95% lower bound/share ${}; "
        "contextual score {} is only the final tie-breaker."
    ).format(
        economics.get("modeled_pop"),
        economics.get("expected_profit_per_share"),
        economics.get("expected_return_on_capital"),
        economics.get("expected_profit_lower_95_per_share"),
        candidate.get("score"),
    )


def run_scan(
    config: Mapping[str, Any],
    token: str,
    as_of: str,
    tickers: Optional[Sequence[str]] = None,
    context_path: Optional[Path] = None,
    offline: bool = False,
    refresh: bool = False,
    max_requests: Optional[int] = None,
    portfolio_nav: Optional[float] = None,
    posture: str = RESEARCH_ONLY,
    use_schwab: bool = True,
    schwab_env_path: Optional[Path] = None,
    client: Optional[OratsClient] = None,
    write_artifacts: bool = True,
    replay_mode: bool = False,
    return_all_candidates: bool = False,
) -> Dict[str, Any]:
    decision_day = date.fromisoformat(as_of)
    if decision_day > date.fromisoformat(today_new_york()):
        raise ValueError("CORAT decision date cannot be in the future")
    output_root, cache_root, state_root = _paths(config)
    configured_items = load_universe(config)
    orats_config = config["orats"]
    if client is None:
        client = OratsClient(
            token=token,
            base_url=str(orats_config["base_url"]),
            cache_root=cache_root,
            state_root=state_root,
            timeout_seconds=float(orats_config["request_timeout_seconds"]),
            max_requests=int(max_requests or orats_config["max_requests_per_run"]),
            monthly_cap=int(orats_config["monthly_request_cap"]),
            requests_per_minute=int(orats_config["requests_per_minute"]),
            offline=offline,
            refresh=refresh,
        )
    traces: List[SourceTrace] = []
    source_errors: List[str] = []
    discovery_cfg = config.get("discovery") if isinstance(config.get("discovery"), Mapping) else {}
    dynamic_discovery = bool(tickers is None and discovery_cfg.get("dynamic_orats_universe", True))
    market_cores: Optional[FetchBundle] = None
    universe_discovery: Dict[str, Any] = {
        "source": "configured ticker subset" if tickers is not None else "configured universe",
        "selected_equities": sum(1 for item in configured_items if item.kind == "equity"),
    }
    if dynamic_discovery:
        market_cores = client.fetch_market_asof("cores", as_of)
        traces.extend(market_cores.traces)
        source_errors.extend(market_cores.errors)
        if market_cores.rows:
            selected_items, universe_discovery = discover_universe(config, market_cores.rows, configured_items)
        else:
            selected_items = configured_items
            universe_discovery["fallback"] = "ORATS complete core universe unavailable; configured universe used"
    else:
        selected_items = load_universe(config, tickers=tickers)
    selected_by_ticker = {item.ticker: item for item in selected_items}
    support_names = supporting_tickers(config, selected_items)
    start_date = (decision_day - timedelta(days=int(config["lookback_calendar_days"]))).isoformat()
    batch_size = int(orats_config["batch_size"])
    dailies = client.fetch_dailies(support_names, start_date, as_of, batch_size=batch_size)
    if market_cores is not None and market_cores.rows:
        cores = market_cores
        ranks = client.fetch_market_asof("ivrank", as_of)
        summaries = client.fetch_market_asof("summaries", as_of)
    else:
        cores = client.fetch_asof("cores", support_names, as_of, batch_size=batch_size)
        ranks = client.fetch_asof("ivrank", support_names, as_of, batch_size=batch_size)
        summaries = client.fetch_asof("summaries", support_names, as_of, batch_size=batch_size)
    traces.extend(dailies.traces + ([] if cores is market_cores else cores.traces) + ranks.traces + summaries.traces)
    source_errors.extend(dailies.errors + ([] if cores is market_cores else cores.errors) + ranks.errors + summaries.errors)
    bars_by_ticker = bars_from_dailies(dailies.rows)
    core_by_ticker = _index_latest(cores.rows)
    volatility_by_ticker = normalize_volatility(support_names, cores.rows, ranks.rows, summaries.rows)
    for ticker, core in core_by_ticker.items():
        bars_by_ticker[ticker] = append_core_spot(bars_by_ticker.get(ticker, []), core, as_of)
    schwab_fresh: Dict[str, bool] = {}
    schwab_status = DATA_UNAVAILABLE
    schwab_cfg = config.get("schwab") if isinstance(config.get("schwab"), Mapping) else {}
    if use_schwab and schwab_cfg and bool(schwab_cfg.get("enabled")) and as_of == today_new_york():
        configured_env = schwab_env_path or Path(str(schwab_cfg.get("env_file") or "/Users/anuppamvi/tradedesk/.env"))
        try:
            schwab_client = SchwabClient(
                configured_env,
                str(schwab_cfg.get("market_data_base_url") or "https://api.schwabapi.com/marketdata/v1"),
                cache_root,
                float(schwab_cfg.get("request_timeout_seconds") or 30),
            )
            schwab_bundle = schwab_client.fetch_quotes(
                support_names,
                int(schwab_cfg.get("quote_batch_size") or 50),
            )
            traces.extend(schwab_bundle.traces)
            source_errors.extend(schwab_bundle.errors)
            for ticker, quote in schwab_bundle.quotes.items():
                quote_bar = quote_to_bar(ticker, quote, as_of)
                if quote_bar is not None:
                    bars_by_ticker[ticker] = merge_quote_bar(bars_by_ticker.get(ticker, []), quote_bar)
                schwab_fresh[ticker] = quote_is_fresh(
                    quote,
                    float(schwab_cfg.get("maximum_quote_age_minutes") or 30),
                )
            if schwab_bundle.quotes:
                schwab_status = "AVAILABLE_FRESH" if any(schwab_fresh.values()) else "AVAILABLE_BUT_STALE"
            else:
                schwab_status = DATA_UNAVAILABLE
        except SchwabError as exc:
            source_errors.append(str(exc))
    snapshots: Dict[str, Any] = {}
    for ticker in support_names:
        vol = volatility_by_ticker.get(ticker, {})
        snapshot = technical_snapshot(ticker, bars_by_ticker.get(ticker, []), as_of, str(vol.get("last_earnings_date") or ""))
        if snapshot is not None:
            snapshots[ticker] = snapshot
    candidate_items = [
        item for item in selected_items
        if item.kind in {"equity", "benchmark", "sector_etf"}
    ]
    candidate_snapshots = [snapshots[item.ticker] for item in candidate_items if item.ticker in snapshots]
    regime = classify_market(snapshots, candidate_snapshots)
    sector_rotation = rank_sectors(
        snapshots,
        sorted(
            set(str(value).upper() for value in config["regime"]["sector_etfs"])
            | {item.sector_etf for item in selected_items if item.sector_etf}
        ),
    )
    context = load_context(context_path, as_of)
    prepared: List[Dict[str, Any]] = []
    liquidity_cfg = config["liquidity"]
    risk_cfg = config["risk"]
    nav = portfolio_nav if portfolio_nav is not None else risk_cfg.get("portfolio_nav")
    for item in candidate_items:
        snapshot = snapshots.get(item.ticker)
        if snapshot is None:
            continue
        sector_snapshot = snapshots.get(item.sector_etf)
        sector_data = sector_rotation.get(item.sector_etf, {"ticker": item.sector_etf, "state": DATA_UNAVAILABLE, "rank": None})
        vol = volatility_by_ticker.get(item.ticker, {"status": DATA_UNAVAILABLE})
        signals = detect_setups(
            snapshot,
            bars_by_ticker.get(item.ticker, []),
            snapshots.get("SPY"),
            sector_snapshot,
            str(sector_data.get("state") or ""),
            str(vol.get("last_earnings_date") or ""),
        )
        setup = signals[0]
        stock_plan = build_stock_plan(snapshot, setup, float(nav) if nav else None, float(risk_cfg["normal_risk_pct"]))
        tcontext = ticker_context(context, item.ticker, as_of)
        preliminary = (
            setup.strength * 60.0
            + max(-10.0, min(10.0, (snapshot.return_20d or 0.0) * 100.0))
            + float(tcontext.get("catalyst_strength") or 0.0) * 15.0
            + (5.0 if setup.triggered else 0.0)
        )
        if snapshot.average_dollar_volume_20d and snapshot.average_dollar_volume_20d >= float(liquidity_cfg["minimum_average_dollar_volume"]):
            preliminary += 5.0
        prepared.append(
            {
                "item": item,
                "snapshot": snapshot,
                "setup": setup,
                "alternate_setups": signals[1:],
                "stock_plan": stock_plan,
                "context": tcontext,
                "volatility": vol,
                "sector": sector_data,
                "preliminary": preliminary,
            }
        )
    prepared.sort(key=lambda row: float(row["preliminary"]), reverse=True)
    context_tickers = set((context.get("tickers") or {}).keys()) if isinstance(context.get("tickers"), dict) else set()
    enrichment_limit = int(config["max_enriched_candidates"])
    prepared_order = [row["item"].ticker for row in prepared]
    triggered_names = [
        row["item"].ticker for row in prepared
        if row["setup"].triggered and row["setup"].direction in {"BULLISH", "BEARISH"} and row["stock_plan"] is not None
    ]
    # Source-backed context and every triggered name are historically enriched.
    # The legacy option limit is retained only for config compatibility; it can
    # never truncate the triggered execution universe.
    option_limit = int(config["max_option_candidates"])
    enriched_names, option_names = _enrichment_plan(
        prepared_order,
        (name for name in context_tickers if name in selected_by_ticker),
        enrichment_limit,
        option_limit,
        triggered_names,
    )
    if not triggered_names:
        option_names = []
    history_cfg = config["history"]
    if option_names and not replay_mode:
        earnings_calendar = fetch_forward_earnings_calendar(
            as_of,
            int(history_cfg["primary_horizon_sessions"]),
            cache_root,
            timeout_seconds=float((config.get("research") or {}).get("request_timeout_seconds") or 15),
            offline=offline,
            refresh=refresh,
        )
        traces.extend(earnings_calendar.traces)
        source_errors.extend(earnings_calendar.errors)
        calendar_complete = not earnings_calendar.errors
        for row in prepared:
            item = row["item"]
            if item.kind != "equity" or item.ticker not in option_names:
                continue
            volatility = row["volatility"]
            scheduled = earnings_calendar.dates_by_ticker.get(item.ticker, "")
            existing = str(volatility.get("next_earnings_date") or "")
            if scheduled and (not existing or scheduled < existing):
                volatility["next_earnings_date"] = scheduled
                volatility["next_earnings_reference"] = "{} (Nasdaq estimated earnings calendar)".format(scheduled)
                volatility["earnings_date_source"] = "NASDAQ EARNINGS CALENDAR (ESTIMATED)"
            if calendar_complete:
                volatility["earnings_calendar_clear_through"] = earnings_calendar.checked_through
                volatility["earnings_calendar_status"] = "CHECKED"
            else:
                volatility["earnings_calendar_status"] = "INCOMPLETE"
    chain_by_ticker: Dict[str, List[Mapping[str, Any]]] = {}
    historical_earnings_by_ticker: Dict[str, List[Mapping[str, Any]]] = {}
    historical_volatility_by_ticker: Dict[str, List[Mapping[str, Any]]] = {}
    for ticker in option_names:
        chain = client.fetch_chain(ticker, as_of, int(orats_config["min_dte"]), int(orats_config["max_dte"]))
        traces.extend(chain.traces)
        source_errors.extend(chain.errors)
        chain_by_ticker[ticker] = list(chain.rows)
        core_history = client.fetch_core_history(ticker, start_date, as_of)
        traces.extend(core_history.traces)
        source_errors.extend(core_history.errors)
        historical_volatility_by_ticker[ticker] = list(core_history.rows)
        item = selected_by_ticker.get(ticker)
        if item is not None and item.kind == "equity":
            earnings_history = client.fetch_earnings(ticker)
            traces.extend(earnings_history.traces)
            source_errors.extend(earnings_history.errors)
            historical_earnings_by_ticker[ticker] = list(earnings_history.rows)
    candidates: List[Dict[str, Any]] = []
    for row in prepared:
        item: UniverseItem = row["item"]
        snapshot = row["snapshot"]
        setup = row["setup"]
        stock_plan = row["stock_plan"]
        if item.ticker in enriched_names and snapshots.get("SPY") is not None:
            history = analyze_analogues(
                setup.name,
                setup.direction,
                bars_by_ticker.get(item.ticker, []),
                bars_by_ticker.get("SPY", []),
                as_of,
                [int(value) for value in history_cfg["forward_horizons"]],
                int(history_cfg["primary_horizon_sessions"]),
                int(history_cfg["minimum_analog_sample"]),
                int(history_cfg["maximum_analog_sample"]),
                int(history_cfg["signal_spacing_sessions"]),
                sector_bars=bars_by_ticker.get(item.sector_etf, []),
                earnings_events=historical_earnings_by_ticker.get(item.ticker, []),
                historical_volatility_rows=historical_volatility_by_ticker.get(item.ticker, []),
                current_iv_hv_ratio=row["volatility"].get("iv_hv_ratio"),
            )
        else:
            history = _empty_history(int(history_cfg["primary_horizon_sessions"]))
        chain_rows = chain_by_ticker.get(item.ticker, [])
        risk_basis_price = float(stock_plan.risk_basis_price or snapshot.price) if stock_plan else snapshot.price
        stop_return = stock_plan.risk_per_share / risk_basis_price if stock_plan and risk_basis_price > 0 else None
        target_return = abs(stock_plan.target_1 - risk_basis_price) / risk_basis_price if stock_plan and risk_basis_price > 0 else None
        current_iv = row["volatility"].get("atm_iv_pct")
        forecast_iv = row["volatility"].get("orats_forecast_iv_20d_pct")
        iv_shift_points = (
            (float(forecast_iv) - float(current_iv)) * min(1.0, float(stock_plan.holding_sessions if stock_plan else 10) / 20.0)
            if current_iv is not None and forecast_iv is not None
            else 0.0
        )
        risk_free_rate = float(row["volatility"].get("risk_free_rate_pct") or 0.0) / 100.0
        dividend_yield = float(row["volatility"].get("dividend_yield_pct") or 0.0) / 100.0
        if stock_plan is not None and item.ticker in option_names:
            option = choose_option_structure(
                chain_rows,
                setup.direction,
                stock_plan.target_1,
                stock_plan.holding_sessions,
                int(liquidity_cfg["minimum_option_open_interest"]),
                int(liquidity_cfg["minimum_option_volume"]),
                float(liquidity_cfg["maximum_option_spread_pct"]),
                scenario_returns=history.primary_returns,
                commission_per_contract=float((config.get("execution") or {}).get("commission_per_contract") or 0.65),
                scenario_paths=history.primary_paths,
                stop_return=stop_return,
                target_return=target_return,
                scenario_adverse_paths=history.primary_adverse_paths,
                scenario_favorable_paths=history.primary_favorable_paths,
                exit_iv_shift_points=iv_shift_points,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
        else:
            option = _empty_option("No current trigger required an exact option-chain comparison, or no stock risk plan existed.")
        comparison_start = option.selection_train_size if option.selection_test_size > 0 else 0
        stock_economics = model_stock_economics(snapshot, stock_plan, history, evaluation_start=comparison_start)
        full_stock_economics = model_stock_economics(snapshot, stock_plan, history)
        option_economics = evaluate_option_evidence(
            option,
            snapshot.price,
            setup.direction,
            stock_plan.holding_sessions if stock_plan else int(history_cfg["primary_horizon_sessions"]),
            history.primary_returns,
            float((config.get("execution") or {}).get("commission_per_contract") or 0.65),
            history.primary_paths,
            stop_return,
            target_return,
            history.primary_adverse_paths,
            history.primary_favorable_paths,
            iv_shift_points,
            risk_free_rate,
            dividend_yield,
        )
        vehicle, vehicle_reason = choose_vehicle(
            stock_plan,
            option,
            row["volatility"],
            as_of=as_of,
            require_earnings_date=bool(config["actionability"]["require_earnings_date_for_options"]),
            stock_economics=stock_economics,
            option_economics=option_economics,
            earnings_applicable=item.kind == "equity",
        ) if stock_plan else ("NO TRADE", "No defensible stock risk plan exists.")
        selected_economics = option_economics if vehicle == "OPTIONS" else stock_economics
        planned_risk = float(nav) * float(risk_cfg["normal_risk_pct"]) if nav else None
        if vehicle == "OPTIONS" and option.maximum_loss and planned_risk:
            sized_units = max(0, int(planned_risk / option.maximum_loss))
            position_sizing = {
                "basis": "maximum option loss",
                "risk_dollars": planned_risk,
                "units": sized_units or None,
                "planned_maximum_loss": (sized_units * option.maximum_loss) if sized_units else None,
            }
        elif vehicle == "STOCK" and stock_plan:
            position_sizing = {
                "basis": "technical invalidation risk per share",
                "risk_dollars": stock_plan.portfolio_risk_dollars,
                "units": stock_plan.units,
                "planned_maximum_loss": stock_plan.maximum_loss,
            }
        else:
            position_sizing = {"basis": DATA_UNAVAILABLE, "risk_dollars": None, "units": None, "planned_maximum_loss": None}
        # ORATS EOD/delayed data can complete a research snapshot but is not a
        # session-time broker reprice. A future live quote adapter may set this
        # true; this ORATS-only release deliberately cannot.
        current_price_repriced = bool(schwab_fresh.get(item.ticker))
        scored = score_candidate(
            snapshot,
            setup,
            stock_plan,
            option,
            vehicle,
            row["volatility"],
            row["context"],
            history,
            row["sector"],
            str(regime.get("label")),
            float(liquidity_cfg["minimum_stock_price"]),
            float(liquidity_cfg["minimum_average_dollar_volume"]),
            float(risk_cfg["minimum_reward_risk"]),
            int(config["actionability"]["minimum_score"]),
            bool(config["actionability"]["require_catalyst_evidence"]),
            bool(config["actionability"].get("require_historical_evidence", True)),
            bool(config["actionability"]["require_earnings_date_for_options"]),
            current_price_repriced,
            selected_economics,
            earnings_applicable=item.kind == "equity",
        )
        candidate = {
            "ticker": item.ticker,
            "name": item.name,
            "sector": item.sector,
            "theme": item.theme,
            "kind": item.kind,
            "sector_etf": item.sector_etf,
            "regime": regime.get("label"),
            "sector_rotation": row["sector"],
            "technical": snapshot.to_dict(),
            "setup": setup.to_dict(),
            "alternate_setups": [value.to_dict() for value in row["alternate_setups"]],
            "stock_plan": stock_plan.to_dict() if stock_plan else {},
            "vehicle": vehicle,
            "vehicle_reason": vehicle_reason,
            "position_sizing": position_sizing,
            "option": option.to_dict(),
            "economics": selected_economics,
            "stock_economics": stock_economics,
            "stock_economics_full_sample": full_stock_economics,
            "option_economics": option_economics,
            "volatility": row["volatility"],
            "context": row["context"],
            "event_risks": event_risks(
                context,
                row["context"],
                as_of,
                stock_plan.holding_sessions if stock_plan else 10,
            ),
            "history": history.to_dict(),
            "positioning": _positioning(chain_rows, row["context"], row["volatility"]),
            "score": scored["score"],
            "components": scored["components"],
            "status": scored["status"],
            "confidence": scored["confidence"],
            "hard_rejections": scored["hard_rejections"],
            "blockers": scored["blockers"],
            "review_notes": scored["notes"],
            "earnings_crossed": scored["earnings_crossed"],
            "portfolio_conflict": False,
            "option_chain_requested": item.ticker in option_names,
            "option_chain_rows": len(chain_rows),
        }
        candidate["primary_risk"] = _primary_risk(candidate)
        candidate["ranking_reason"] = _ranking_reason(candidate)
        candidates.append(candidate)
    status_rank = {"TARGET TRADE": 4, "SETUP ONLY — NOT A TRADE": 3, "NO TRADE — EDGE NOT POSITIVE": 2, "WATCHLIST": 1, "REJECTED / AVOID": 0}
    candidates.sort(key=lambda item: (status_rank.get(item["status"], -1),) + _economics_rank(item), reverse=True)
    _apply_correlation_caps(candidates, int(risk_cfg.get("maximum_correlated_ideas") or 2))
    candidates.sort(key=lambda item: (status_rank.get(item["status"], -1),) + _economics_rank(item), reverse=True)
    all_candidates = list(candidates)
    candidate_audit = [
        {
            "rank": index,
            "ticker": candidate["ticker"],
            "kind": candidate["kind"],
            "setup": candidate["setup"].get("name"),
            "direction": candidate["setup"].get("direction"),
            "triggered": candidate["setup"].get("triggered"),
            "history_sample": candidate["history"].get("sample_size"),
            "option_chain_requested": candidate.get("option_chain_requested"),
            "option_chain_rows": candidate.get("option_chain_rows"),
            "option_structures_evaluated": candidate["option"].get("candidate_count"),
            "option_strategy": candidate["option"].get("strategy"),
            "option_holdout_sample": candidate["option_economics"].get("holdout_sample_size"),
            "option_pop": candidate["option_economics"].get("modeled_pop"),
            "option_expected_profit": candidate["option_economics"].get("expected_profit_dollars"),
            "option_stress_floor": candidate["option_economics"].get("robust_expected_profit_dollars"),
            "stock_pop": candidate["stock_economics"].get("modeled_pop"),
            "stock_expected_profit_per_share": candidate["stock_economics"].get("expected_profit_per_share"),
            "vehicle": candidate.get("vehicle"),
            "vehicle_reason": candidate.get("vehicle_reason"),
            "selected_pop": candidate["economics"].get("modeled_pop"),
            "selected_expected_profit": (
                candidate["economics"].get("expected_profit_dollars")
                if candidate.get("vehicle") == "OPTIONS"
                else candidate["economics"].get("expected_profit_per_share")
            ),
            "selected_expected_profit_unit": (
                "DOLLARS_PER_1_LOT"
                if candidate.get("vehicle") == "OPTIONS"
                else "DOLLARS_PER_SHARE"
            ),
            "selected_expected_profit_lower_95": (
                candidate["economics"].get("expected_profit_lower_95_dollars")
                if candidate.get("vehicle") == "OPTIONS"
                else candidate["economics"].get("expected_profit_lower_95_per_share")
            ),
            "selected_evidence_sample": candidate["economics"].get("model_sample_size"),
            "earnings_reference": candidate["volatility"].get("next_earnings_reference")
            or candidate["volatility"].get("next_earnings_date"),
            "earnings_crossed": candidate.get("earnings_crossed"),
            "ranking_reason": candidate.get("ranking_reason"),
            "primary_risk": candidate.get("primary_risk"),
            "blockers": candidate.get("blockers"),
            "hard_rejections": candidate.get("hard_rejections"),
            "review_notes": candidate.get("review_notes"),
            "option_reasons": candidate["option"].get("reasons"),
            "status": candidate.get("status"),
            "score": candidate.get("score"),
        }
        for index, candidate in enumerate(all_candidates, start=1)
    ]
    final_limit = int(config["max_final_ideas"])
    candidates = candidates if return_all_candidates else candidates[:final_limit]
    positive_option_alternatives = [
        candidate for candidate in all_candidates
        if candidate["option_economics"].get("expected_profit_dollars") is not None
        and float(candidate["option_economics"]["expected_profit_dollars"]) > 0
    ]
    diagnostics = {
        "configured_universe_rows": len(configured_items),
        "discovery_selected_rows": len(selected_items),
        "initially_scanned": len(prepared),
        "missing_technical_history": len(candidate_items) - len(prepared),
        "triggered_setups": len(triggered_names),
        "history_requested_for_triggered": len(triggered_names),
        "history_with_samples_for_triggered": sum(
            1 for candidate in all_candidates
            if candidate["setup"].get("triggered")
            and int(candidate["history"].get("sample_size") or 0) > 0
        ),
        "option_chains_requested": len(option_names),
        "option_chains_with_rows": sum(1 for ticker in option_names if chain_by_ticker.get(ticker)),
        "option_structures_evaluated": sum(int(candidate["option"].get("candidate_count") or 0) for candidate in all_candidates),
        "passing_liquidity": sum(1 for row in prepared if (row["snapshot"].average_dollar_volume_20d or 0) >= float(liquidity_cfg["minimum_average_dollar_volume"])),
        "passing_technical_setup": sum(1 for row in prepared if row["setup"].name != "NO QUALIFYING SETUP"),
        "passing_catalyst_regime": sum(
            1 for candidate in all_candidates
            if any(
                str(row.get("direction") or "").upper() == str(candidate["setup"].get("direction") or "")
                for row in candidate["context"].get("actionable_catalysts") or []
            ) and candidate["components"]["market_regime_alignment"] >= 3
        ),
        "passing_risk_reward": sum(1 for candidate in all_candidates if candidate["stock_plan"] and candidate["stock_plan"].get("reward_risk_2", 0) >= float(risk_cfg["minimum_reward_risk"])),
        "rejected_for_earnings": sum(
            1 for candidate in positive_option_alternatives
            if "earnings" in str(candidate.get("vehicle_reason") or "").lower()
        ),
        "rejected_for_options": sum(
            1 for candidate in all_candidates
            if candidate["setup"].get("triggered")
            and candidate.get("stock_plan")
            and not candidate["option"].get("valid")
        ),
        "positive_option_alternatives": len(positive_option_alternatives),
        "negative_or_unavailable_option_alternatives": sum(
            1 for candidate in all_candidates
            if candidate["setup"].get("triggered") and candidate not in positive_option_alternatives
        ),
        "target_trades": sum(1 for candidate in all_candidates if candidate["status"] == TARGET_TRADE),
        "option_target_trades": sum(
            1 for candidate in all_candidates
            if candidate["status"] == TARGET_TRADE and candidate.get("vehicle") == "OPTIONS"
        ),
        "stock_target_trades": sum(
            1 for candidate in all_candidates
            if candidate["status"] == TARGET_TRADE and candidate.get("vehicle") == "STOCK"
        ),
        "finally_qualifying": sum(1 for candidate in all_candidates if candidate["status"] == TARGET_TRADE),
        "displayed_ideas": len(candidates),
        "displayed_target_trades": sum(1 for candidate in candidates if candidate["status"] == TARGET_TRADE),
        "universe_discovery": universe_discovery,
    }
    source_traces = _dedupe_traces(traces)
    latest_price_date = max((candidate["technical"].get("price_date") or "" for candidate in candidates), default="")
    latest_option_date = max((candidate["option"].get("quote_trade_date") or "" for candidate in candidates), default="")
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "posture": posture,
        "as_of": as_of,
        "generated_at_utc": utc_now(),
        "data_cutoff": "Price {} / option {}".format(latest_price_date or DATA_UNAVAILABLE, latest_option_date or DATA_UNAVAILABLE),
        "timestamp_note": "ORATS price and option timestamps are shown on every ticket. Schwab is optional market data only (status {}; {} fresh candidate quote(s)); selection uses the completed ORATS as-of data and the displayed limit.".format(
            schwab_status,
            sum(1 for ticker in selected_by_ticker if schwab_fresh.get(ticker)),
        ),
        "regime": regime,
        "sector_rotation": sector_rotation,
        "candidates": candidates,
        "candidate_audit": candidate_audit,
        "universe_discovery": universe_discovery,
        "diagnostics": diagnostics,
        "source_traces": source_traces,
        "source_errors": sorted(set(source_errors)),
        "orats_usage": client.usage(),
        "schwab": {
            "status": schwab_status,
            "fresh_tickers": sorted(ticker for ticker, fresh in schwab_fresh.items() if fresh),
            "read_only": True,
            "shared_token_mutated": False,
        },
        "context": {
            "status": context.get("status"),
            "source_path": context.get("source_path"),
            "source_sha256": context.get("source_sha256"),
            "reason": context.get("reason"),
            "market_events": context.get("market_events") or [],
            "research_metadata": context.get("research_metadata") or {},
        },
    }
    if not write_artifacts:
        result["artifacts"] = {}
        return result
    run_digest = sha256_bytes(canonical_json({"as_of": as_of, "candidates": candidates, "generated": result["generated_at_utc"]}).encode("utf-8"))[:12]
    run_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), run_digest)
    run_dir = output_root / as_of / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "corat-{}.md".format(as_of)
    board_path = run_dir / "board.csv"
    run_path = run_dir / "run.json"
    diagnostics_path = run_dir / "diagnostics.json"
    audit_path = run_dir / "candidate-audit.json"
    sources_path = run_dir / "sources.json"
    write_text(report_path, render_report(result))
    write_text(board_path, render_board_csv(candidates))
    write_json(run_path, result)
    write_json(diagnostics_path, diagnostics)
    write_json(audit_path, candidate_audit)
    write_json(sources_path, {"traces": source_traces, "errors": result["source_errors"]})
    artifact_paths = [report_path, board_path, run_path, diagnostics_path, audit_path, sources_path]
    manifest = {
        "schema_version": "corat.manifest.v1",
        "run_id": run_id,
        "posture": posture,
        "as_of": as_of,
        "generated_at_utc": result["generated_at_utc"],
        "config_path": config.get("_config_path"),
        "config_sha256": sha256_file(Path(str(config["_config_path"]))) if config.get("_config_path") else "",
        "context_path": context.get("source_path") or "",
        "context_sha256": context.get("source_sha256") or "",
        "inputs": source_traces,
        "outputs": {str(path): sha256_file(path) for path in artifact_paths},
        "secrets_persisted": False,
        "order_submission_surface": False,
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    latest_path = output_root / as_of / "latest.json"
    write_json(latest_path, {"run_id": run_id, "run_dir": str(run_dir), "run_path": str(run_path), "report_path": str(report_path), "manifest_path": str(manifest_path)})
    result["artifacts"] = {
        "run_dir": str(run_dir),
        "report": str(report_path),
        "board": str(board_path),
        "run": str(run_path),
        "diagnostics": str(diagnostics_path),
        "candidate_audit": str(audit_path),
        "sources": str(sources_path),
        "manifest": str(manifest_path),
        "latest": str(latest_path),
    }
    return result


def compare_runs(previous: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    before = {item["ticker"]: item for item in previous.get("candidates") or []}
    after = {item["ticker"]: item for item in current.get("candidates") or []}
    changes = []
    for ticker in sorted(set(before) | set(after)):
        old = before.get(ticker)
        new = after.get(ticker)
        if old is None:
            changes.append({"ticker": ticker, "change": "NEW", "detail": "Entered top board at score {} / {}".format(new.get("score"), new.get("status"))})
            continue
        if new is None:
            changes.append({"ticker": ticker, "change": "REMOVED", "detail": "Left the top board; previous score {} / {}".format(old.get("score"), old.get("status"))})
            continue
        details = []
        if old.get("status") != new.get("status"):
            details.append("status {} -> {}".format(old.get("status"), new.get("status")))
        if old.get("score") != new.get("score"):
            details.append("score {} -> {}".format(old.get("score"), new.get("score")))
        old_setup = (old.get("setup") or {}).get("name")
        new_setup = (new.get("setup") or {}).get("name")
        if old_setup != new_setup:
            details.append("setup {} -> {}".format(old_setup, new_setup))
        old_price = float((old.get("technical") or {}).get("price") or 0)
        new_price = float((new.get("technical") or {}).get("price") or 0)
        if old_price > 0 and new_price > 0:
            change = new_price / old_price - 1.0
            if abs(change) >= 0.002:
                details.append("price {:+.2%}".format(change))
        if details:
            changes.append({"ticker": ticker, "change": "MATERIAL", "detail": "; ".join(details)})
    return {
        "schema_version": "corat.delta.v1",
        "previous_as_of": previous.get("as_of"),
        "current_as_of": current.get("as_of"),
        "changes": changes,
    }


def render_delta(delta: Mapping[str, Any]) -> str:
    lines = [
        "# CORAT Delta Scan — {} vs {}".format(delta.get("current_as_of"), delta.get("previous_as_of")),
        "",
        "Only material board changes are shown. A changed score is not an order authorization.",
        "",
    ]
    if not delta.get("changes"):
        lines.append("NO MATERIAL CHANGE.")
    else:
        lines.extend("- **{} — {}**: {}".format(row.get("ticker"), row.get("change"), row.get("detail")) for row in delta.get("changes") or [])
    return "\n".join(lines) + "\n"
