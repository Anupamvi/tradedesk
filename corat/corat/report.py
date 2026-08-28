"""Human-readable CORAT decision-board rendering."""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from corat.constants import DATA_UNAVAILABLE, MANUAL_ONLY, NO_POSITIVE_EDGE, REJECTED, SETUP_ONLY, TARGET_TRADE, WATCHLIST


def _money(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DATA_UNAVAILABLE
    if not math.isfinite(number):
        return DATA_UNAVAILABLE
    return "${:,.{digits}f}".format(number, digits=digits)


def _pct(value: Any, digits: int = 1, already_percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DATA_UNAVAILABLE
    if not math.isfinite(number):
        return DATA_UNAVAILABLE
    if not already_percent:
        number *= 100.0
    return "{:.{digits}f}%".format(number, digits=digits)


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DATA_UNAVAILABLE
    if not math.isfinite(number):
        return "inf" if number > 0 else DATA_UNAVAILABLE
    return "{:,.{digits}f}".format(number, digits=digits)


def _text(value: Any) -> str:
    if value in (None, "", []):
        return DATA_UNAVAILABLE
    return str(value).replace("\n", " ").strip()


def _table_escape(value: Any) -> str:
    return _text(value).replace("|", "\\|")


def _all_reasons(candidate: Mapping[str, Any]) -> List[str]:
    return list(candidate.get("hard_rejections") or []) + list(candidate.get("blockers") or [])


def _idea_row(rank: int, candidate: Mapping[str, Any]) -> str:
    setup = candidate.get("setup") or {}
    plan = candidate.get("stock_plan") or {}
    vol = candidate.get("volatility") or {}
    economics = candidate.get("economics") or {}
    expected_profit = economics.get("expected_profit_dollars") if candidate.get("vehicle") == "OPTIONS" else economics.get("expected_profit_per_share")
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} sessions | {} | {} |".format(
        rank,
        candidate.get("ticker"),
        setup.get("direction"),
        candidate.get("vehicle"),
        _table_escape(setup.get("name")),
        "{}-{}".format(_money(plan.get("entry_low")), _money(plan.get("entry_high"))),
        _money(plan.get("stop")),
        _money(plan.get("target_2")),
        _pct(economics.get("modeled_pop")),
        _money(expected_profit),
        plan.get("holding_sessions") or DATA_UNAVAILABLE,
        "{}{}".format(_text(vol.get("next_earnings_reference") or vol.get("next_earnings_date")), " (crossed)" if candidate.get("earnings_crossed") else ""),
        candidate.get("status"),
    )


def _option_legs(candidate: Mapping[str, Any]) -> str:
    option = candidate.get("option") or {}
    legs = option.get("legs") or []
    if not legs:
        return DATA_UNAVAILABLE
    return "; ".join(
        "{} {} {} {}".format(leg.get("action"), leg.get("quantity") or 1, leg.get("expiration"), "{} {}".format(leg.get("strike"), leg.get("option_type")))
        for leg in legs
    )


def _target_ticket(candidate: Mapping[str, Any]) -> str:
    economics = candidate.get("economics") or {}
    plan = candidate.get("stock_plan") or {}
    option = candidate.get("option") or {}
    if candidate.get("vehicle") == "OPTIONS":
        entry = "{} limit {}".format(str(option.get("debit_credit") or "DEBIT").lower(), _money(option.get("expected_entry")))
        maximum_loss = _money(option.get("maximum_loss"))
        expected_profit = _money(economics.get("expected_profit_dollars"))
        structure = _option_legs(candidate)
    else:
        entry = "{}–{}".format(_money(plan.get("entry_low")), _money(plan.get("entry_high")))
        sizing = candidate.get("position_sizing") or {}
        units = sizing.get("units")
        sized_maximum_loss = sizing.get("planned_maximum_loss")
        maximum_loss = (
            _money(sized_maximum_loss)
            if sized_maximum_loss is not None
            else "{} / share to stop".format(_money(plan.get("risk_per_share")))
        )
        expected_profit = (
            "{} at shown size ({} / share)".format(
                _money(economics.get("expected_position_profit")),
                _money(economics.get("expected_profit_per_share")),
            )
            if economics.get("expected_position_profit") is not None
            else "{} / share".format(_money(economics.get("expected_profit_per_share")))
        )
        structure = "{} shares".format(units) if units is not None else "Shares = risk budget / risk per share"
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        candidate.get("ticker"),
        (candidate.get("setup") or {}).get("direction"),
        candidate.get("vehicle"),
        _table_escape(structure),
        entry,
        _money(plan.get("stop")),
        _money(plan.get("target_1")),
        _money(plan.get("target_2")),
        _pct(economics.get("modeled_pop")),
        expected_profit,
        maximum_loss,
        "{} sessions / N={}".format(
            _text(plan.get("holding_sessions")),
            _text(economics.get("model_sample_size")),
        ),
    )


def _source_lines(candidate: Mapping[str, Any]) -> List[str]:
    lines = []
    context = candidate.get("context") or {}
    for family in ("catalysts", "events", "x_intelligence", "options_flow"):
        for row in context.get(family) or []:
            lines.append(
                "- {} — {} — {} — {} — {}".format(
                    _text(row.get("source")),
                    _text(row.get("published_at")),
                    _text(row.get("classification")),
                    _text(row.get("title") or row.get("claim")),
                    _text(row.get("source_url")),
                )
            )
    if not lines:
        lines.append("- {} — no structured catalyst/news/X/flow evidence was supplied for this ticker.".format(DATA_UNAVAILABLE))
    return lines


def _greek_explanation(option: Mapping[str, Any], holding_sessions: int) -> List[str]:
    if not option.get("valid"):
        return ["- Greeks: {} because no exact option structure passed.".format(DATA_UNAVAILABLE)]
    delta = option.get("delta")
    gamma = option.get("gamma")
    theta = option.get("theta")
    vega = option.get("vega")
    theta_number = float(theta) if theta is not None else 0.0
    theta_effect = "time decay favors the position" if theta_number > 0 else "time decay works against the position" if theta_number < 0 else "modeled net time decay is neutral"
    return [
        "- Delta {}: approximate spread-price sensitivity to a $1 underlying move; it is directional exposure, not win probability.".format(_num(delta, 3)),
        "- Gamma {}: approximate change in net delta per $1 underlying move; the selected 21-75 DTE window avoids routine ultra-short gamma.".format(_num(gamma, 4)),
        "- Theta {} per day per share (about {} over {} sessions per position if unchanged): {}.".format(_num(theta, 4), _money(option.get("theta_holding_cost")), holding_sessions, theta_effect),
        "- Vega {}: a one-vol-point IV change is expected to change spread value in the same sign; IV compression is a risk when net vega is positive.".format(_num(vega, 4)),
    ]


def render_candidate(candidate: Mapping[str, Any]) -> str:
    technical = candidate.get("technical") or {}
    setup = candidate.get("setup") or {}
    plan = candidate.get("stock_plan") or {}
    volatility = candidate.get("volatility") or {}
    option = candidate.get("option") or {}
    history = candidate.get("history") or {}
    context = candidate.get("context") or {}
    sector = candidate.get("sector_rotation") or {}
    positioning = candidate.get("positioning") or {}
    sizing = candidate.get("position_sizing") or {}
    economics = candidate.get("economics") or {}
    option_economics = candidate.get("option_economics") or {}
    event_risk = candidate.get("event_risks") or []
    entry_side = str(option.get("debit_credit") or "DEBIT").lower()
    if str(candidate.get("kind") or "equity") != "equity":
        earnings_text = "NOT APPLICABLE — diversified fund/index product / NO"
    elif candidate.get("earnings_crossed"):
        earnings_text = "{} / YES — REJECT ordinary option".format(_text(volatility.get("next_earnings_reference") or volatility.get("next_earnings_date")))
    elif volatility.get("next_earnings_date") or (volatility.get("weeks_to_next_earnings") is not None and float(volatility.get("weeks_to_next_earnings") or 0) > 0):
        earnings_text = "{} / NO based on planned horizon".format(_text(volatility.get("next_earnings_reference") or volatility.get("next_earnings_date")))
    elif volatility.get("earnings_calendar_clear_through"):
        earnings_text = "No scheduled report found through {} in the checked Nasdaq estimated calendar / NO".format(volatility.get("earnings_calendar_clear_through"))
    else:
        earnings_text = "DATA UNAVAILABLE / ordinary option not selected"
    lines = [
        "## {} — {}".format(candidate.get("ticker"), setup.get("direction")),
        "",
        "Status: **{}**  ".format(candidate.get("status")),
        "Security: {}  ".format(candidate.get("name")),
        "Current/delayed price: {}  ".format(_money(technical.get("price"))),
        "Price date/completeness: {} / {}  ".format(_text(technical.get("price_date")), "complete OHLCV" if technical.get("price_complete") else "live/partial observation requiring freshness check"),
        "Price source/update: {} / {}  ".format(_text(technical.get("price_source")), _text(technical.get("price_updated_at"))),
        "Setup: {}  ".format(setup.get("name")),
        "Why now: {}  ".format(setup.get("reason")),
        "Market regime: {}  ".format(candidate.get("regime")),
        "Sector/theme: {} / {}  ".format(candidate.get("sector"), candidate.get("theme")),
        "Sector rotation: {} (rank {})  ".format(_text(sector.get("state")), _text(sector.get("rank"))),
        "Relative strength: 5d {} / 20d {} / 60d {}  ".format(_pct(technical.get("return_5d")), _pct(technical.get("return_20d")), _pct(technical.get("return_60d"))),
        "",
        "Catalyst: {}  ".format(_text((context.get("catalysts") or [{}])[0].get("title") if context.get("catalysts") else DATA_UNAVAILABLE)),
        "Catalyst freshness: {}  ".format(_text((context.get("catalysts") or [{}])[0].get("freshness") if context.get("catalysts") else DATA_UNAVAILABLE)),
        "Known event exposure during planned hold: {}  ".format(
            "; ".join("{} — {}".format(_text(row.get("event_date")), _text(row.get("title") or row.get("claim"))) for row in event_risk)
            or "None found in the supplied sourced event calendar; absence is not proof that no event exists."
        ),
        "",
        "Technical structure: price {} EMA20 {}, SMA50 {}, SMA200 {}.  ".format(_money(technical.get("price")), _money(technical.get("ema20")), _money(technical.get("sma50")), _money(technical.get("sma200"))),
        "AVWAP analysis: {}  ".format(
            "; ".join("{} from {} ({})".format(_money(level.get("value")), level.get("anchor_date"), level.get("anchor_reason")) for level in technical.get("avwaps") or []) or DATA_UNAVAILABLE
        ),
        "Volume: relative volume {}, average dollar volume {}.  ".format(_num(technical.get("relative_volume_20d")), _money(technical.get("average_dollar_volume_20d"), 0)),
        "Important support/resistance: {} / {}.  ".format(_money(technical.get("support")), _money(technical.get("resistance"))),
        "",
        "### Trade",
        "",
        "Vehicle: **{}** — {}  ".format(candidate.get("vehicle"), candidate.get("vehicle_reason")),
        "Entry zone: {} to {}  ".format(_money(plan.get("entry_low")), _money(plan.get("entry_high"))),
        "Trigger: {}  ".format(_text(plan.get("trigger"))),
        "Thesis-aligned stop: {}  ".format(_money(plan.get("stop"))),
        "Setup invalidation rule: {}  ".format(_text(setup.get("invalidation"))),
        "Target 1 / Target 2: {} / {}  ".format(_money(plan.get("target_1")), _money(plan.get("target_2"))),
        "Expected holding period: {} sessions  ".format(_text(plan.get("holding_sessions"))),
        "Risk/reward to Target 1 / 2: {} / {}  ".format(_num(plan.get("reward_risk_1")), _num(plan.get("reward_risk_2"))),
        "Risk per share from worst permitted entry {}: {}; planned portfolio risk: {}; units: {}.  ".format(_money(plan.get("risk_basis_price")), _money(plan.get("risk_per_share")), _money(plan.get("portfolio_risk_dollars")), _text(plan.get("units"))),
        "Selected-vehicle sizing basis / units / planned maximum loss: {} / {} / {}.  ".format(_text(sizing.get("basis")), _text(sizing.get("units")), _money(sizing.get("planned_maximum_loss"))),
        "Modeled POP / sample: {} / N={}  ".format(_pct(economics.get("modeled_pop")), _text(economics.get("model_sample_size"))),
        "Expected profit: {}  ".format(
            _money(economics.get("expected_profit_dollars"))
            if candidate.get("vehicle") == "OPTIONS"
            else "{} per share{}".format(
                _money(economics.get("expected_profit_per_share")),
                " / {} at displayed size".format(_money(economics.get("expected_position_profit"))) if economics.get("expected_position_profit") is not None else "",
            )
        ),
        "Expected-profit 95% interval: {} to {}  ".format(
            _money(economics.get("expected_profit_lower_95_dollars") if candidate.get("vehicle") == "OPTIONS" else economics.get("expected_profit_lower_95_per_share")),
            _money(economics.get("expected_profit_upper_95_dollars") if candidate.get("vehicle") == "OPTIONS" else economics.get("expected_profit_upper_95_per_share")),
        ),
        "Profit factor: {}  ".format(_num(economics.get("profit_factor"))),
        "POP/expected-profit method: {}  ".format(_text(economics.get("method"))),
        "",
        "### Options comparison",
        "",
        "Strategy: {}  ".format(_text(option.get("strategy"))),
        "Expiration / DTE: {} / {}  ".format(_text(option.get("expiration")), _text(option.get("dte"))),
        "Legs: {}  ".format(
            "; ".join("{} {} {} {} @ bid {} x {} / ask {} x {}".format(leg.get("action"), leg.get("quantity"), leg.get("option_type"), _money(leg.get("strike")), _money(leg.get("bid")), _text(leg.get("bid_size")), _money(leg.get("ask")), _text(leg.get("ask_size"))) for leg in option.get("legs") or []) or DATA_UNAVAILABLE
        ),
        "Options liquidity: {}  ".format(
            "; ".join("{} {} OI {} / volume {} / spread {}".format(leg.get("option_type"), _money(leg.get("strike")), _text(leg.get("open_interest")), _text(leg.get("volume")), _pct(leg.get("spread_pct"))) for leg in option.get("legs") or []) or DATA_UNAVAILABLE
        ),
        "Bid/ask execution gate: {}  ".format("PASSED" if option.get("valid") else "FAILED / NOT EVALUATED"),
        "Midpoint / expected {} / natural {}: {} / {} / {}  ".format(entry_side, entry_side, _money(option.get("midpoint_entry")), _money(option.get("expected_entry")), _money(option.get("natural_entry"))),
        "Maximum loss / maximum gain / breakeven: {} / {} / {}  ".format(_money(option.get("maximum_loss")), _money(option.get("maximum_gain")), _money(option.get("breakeven"))),
    ]
    lines.extend(_greek_explanation(option, int(plan.get("holding_sessions") or 0)))
    lines.extend(
        [
            "",
            "ATM IV / IV Rank / IV Percentile: {} / {} / {}  ".format(_pct(volatility.get("atm_iv_pct"), already_percent=True), _num(volatility.get("iv_rank_1y")), _num(volatility.get("iv_percentile_1y"))),
            "HV20 / ex-earnings IV20 / ORATS forecast realized vol20: {} / {} / {}  ".format(_pct(volatility.get("historical_volatility_20d_pct"), already_percent=True), _pct(volatility.get("ex_earnings_iv_20d_pct"), already_percent=True), _pct(volatility.get("orats_forecast_realized_20d_pct"), already_percent=True)),
            "IV/HV / IV/forecast: {} / {}  ".format(_num(volatility.get("iv_hv_ratio")), _num(volatility.get("iv_forecast_realized_ratio"))),
            "Skew / term structure / ORATS confidence: {} / {} / {}  ".format(_num(volatility.get("skew")), _text(volatility.get("term_structure")), _num(volatility.get("orats_confidence"))),
            "CORAT modeled option POP / expected profit / expected return on max loss: {} / {} / {}  ".format(_pct(option_economics.get("modeled_pop")), _money(option_economics.get("expected_profit_dollars")), _pct(option_economics.get("expected_return_on_max_loss"))),
            "Structure selection: {}  ".format(_text(option.get("selection_method"))),
            "Candidates / train paths / held-out paths: {} / {} / {}  ".format(_text(option_economics.get("selection_candidate_count")), _text(option_economics.get("selection_train_size")), _text(option_economics.get("holdout_sample_size"))),
            "Option POP sample / profit factor / modeled exit friction / commission: N={} / {} / {} / {}  ".format(_text(option_economics.get("model_sample_size")), _num(option_economics.get("profit_factor")), _money(option_economics.get("estimated_exit_slippage")), _money(option_economics.get("round_trip_commission"))),
            "Expected-profit 95% interval: {} to {}  ".format(_money(option_economics.get("expected_profit_lower_95_dollars")), _money(option_economics.get("expected_profit_upper_95_dollars"))),
            "Natural-fill / IV -2pt / IV +2pt expected profit: {} / {} / {}  ".format(_money(option_economics.get("natural_fill_expected_profit_dollars")), _money(option_economics.get("iv_down_2_expected_profit_dollars")), _money(option_economics.get("iv_up_2_expected_profit_dollars"))),
            "Stress floor / positive across displayed stresses: {} / {}  ".format(_money(option_economics.get("robust_expected_profit_dollars")), option_economics.get("robust_positive_across_fill_iv_stress")),
            "Modeled exit IV / ORATS forecast shift: {} / {} points  ".format(_pct(option_economics.get("exit_iv_pct"), already_percent=True), _num(option_economics.get("iv_shift_points"))),
            "POP source: CORAT scenario model, not an ORATS POP field. {}  ".format(_text(option_economics.get("method"))),
            "ORATS theoretical structure value / edge: {} / {}  ".format(_money(option.get("orats_theoretical_value")), _money(option.get("theoretical_edge"))),
            "Earnings date / held through earnings: {}  ".format(earnings_text),
            "Option quote date/update: {} / {}  ".format(_text(option.get("quote_trade_date")), _text(option.get("quote_updated_at"))),
            "Option execution notes: {}  ".format("; ".join(option.get("reasons") or []) or "Two-sided quotes available; OI, volume, width, modeled friction, and exact limit are displayed for review." if option.get("valid") else "No exact option structure passed."),
            "",
            "### Positioning and flow",
            "",
            "Major call OI levels: {}  ".format(_text(positioning.get("major_call_oi_levels"))),
            "Major put OI levels: {}  ".format(_text(positioning.get("major_put_oi_levels"))),
            "Gamma context: {}  ".format(_text(positioning.get("gamma_context"))),
            "Options flow: {}  ".format(_text(positioning.get("flow_summary"))),
            "",
            "### X intelligence",
            "",
            "Sentiment/developments: {}  ".format(_text([row.get("claim") or row.get("title") for row in context.get("x_intelligence") or []])),
            "Mention acceleration: {}  ".format(_text(context.get("mention_acceleration"))),
            "Spam/pump risk: {}  ".format("PRESENT" if context.get("x_spam_risk") else "not identified from supplied evidence" if context.get("x_intelligence") else DATA_UNAVAILABLE),
            "",
            "### Historical validation",
            "",
            "Method: {}  ".format(_text(history.get("method"))),
            "Comparable setup sample: {} (reliable threshold met: {})  ".format(_text(history.get("sample_size")), history.get("reliable")),
            "Matched dimensions: {}  ".format(", ".join(history.get("similarity_dimensions") or []) or DATA_UNAVAILABLE),
            "Unavailable historical dimensions: {}  ".format(", ".join(history.get("missing_dimensions") or []) or "none"),
            "Historical win rate / expectancy: {} / {}  ".format(_pct(history.get("win_rate")), _pct(history.get("expectancy"))),
            "Average winner / loser: {} / {}  ".format(_pct(history.get("average_winner")), _pct(history.get("average_loser"))),
            "Profit factor / MAE / MFE / max drawdown: {} / {} / {} / {}  ".format(_num(history.get("profit_factor")), _pct(history.get("mae")), _pct(history.get("mfe")), _pct(history.get("max_drawdown"))),
            "",
            "### Final",
            "",
            "Trade score: **{}/100** (ranking quality, not win probability)  ".format(candidate.get("score")),
            "Confidence: **{}**  ".format(candidate.get("confidence")),
            "Blockers: {}  ".format("; ".join(candidate.get("blockers") or []) or "none"),
            "Review notes: {}  ".format("; ".join(candidate.get("review_notes") or []) or "none"),
            "Hard rejections: {}  ".format("; ".join(candidate.get("hard_rejections") or []) or "none"),
            "Primary risk: {}  ".format(_text(candidate.get("primary_risk"))),
            "Why ranked here: {}  ".format(_text(candidate.get("ranking_reason"))),
            "",
            "Source traceability:",
            "",
        ]
    )
    lines.extend(_source_lines(candidate))
    return "\n".join(lines)


def render_report(result: Mapping[str, Any]) -> str:
    candidates = list(result.get("candidates") or [])
    targets = [item for item in candidates if item.get("status") == TARGET_TRADE]
    diagnostics = result.get("diagnostics") or {}
    total_targets = int(diagnostics.get("target_trades") or len(targets))
    total_option_targets = int(
        diagnostics.get("option_target_trades")
        if diagnostics.get("option_target_trades") is not None
        else sum(1 for item in targets if item.get("vehicle") == "OPTIONS")
    )
    total_stock_targets = int(
        diagnostics.get("stock_target_trades")
        if diagnostics.get("stock_target_trades") is not None
        else sum(1 for item in targets if item.get("vehicle") == "STOCK")
    )
    lead = (
        "{} TARGET TRADE{} IDENTIFIED FOR MANUAL REVIEW — {} OPTIONS / {} STOCK.".format(
            total_targets,
            "S" if total_targets != 1 else "",
            total_option_targets,
            total_stock_targets,
        )
        if total_targets
        else "NO POSITIVE-EXPECTANCY TARGET TRADE WAS IDENTIFIED FROM THE AVAILABLE EVIDENCE."
    )
    lines = [
        "# CORAT Swing-Research Board — {}".format(result.get("as_of")),
        "",
        "**{}**".format(lead),
        "",
        "**{}**. CORAT has no order-submission capability.".format(MANUAL_ONLY),
        "",
        "Run posture: **{}**  ".format(result.get("posture")),
        "Data cutoff: {}  ".format(result.get("data_cutoff")),
        "Price/option timestamp note: {}  ".format(result.get("timestamp_note")),
        "Selection rule: exact trade plan, present setup trigger, and positive modeled expected profit. Target trades are ranked by expected return, expected profit, POP, uncertainty/stress evidence, and sample size; the 0–100 context score is only a tie-breaker.  ",
        (
            "Display scope: top {} of {} qualifying targets; the immutable candidate audit contains every scanned name and disposition.  ".format(
                len(targets), total_targets
            )
            if total_targets > len(targets)
            else "Display scope: all {} qualifying targets are shown.  ".format(total_targets)
        ),
        "",
        "## Target trades",
        "",
        "| Ticker | Direction | Vehicle | Exact position | Entry/limit | Stop | Target 1 | Target 2 | Modeled POP | Expected profit | Max loss | Horizon/sample |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if targets:
        lines.extend(_target_ticket(candidate) for candidate in targets)
    else:
        lines.append("| None | — | — | — | — | — | — | — | — |")
    research = (result.get("context") or {}).get("research_metadata") or {}
    if research:
        lines.extend(
            [
                "",
                "Automatic research: {} ticker(s), {} dated headline row(s). {}".format(
                    len(research.get("researched_tickers") or []),
                    research.get("evidence_rows_added_or_seen") or 0,
                    _text(research.get("x_status")),
                ),
            ]
        )
    lines.extend([
        "",
        "## Market regime",
        "",
        "Classification: **{}**  ".format((result.get("regime") or {}).get("label")),
        "Strategy bias: {}  ".format((result.get("regime") or {}).get("strategy_bias")),
        "Breadth sample/reliable: {} / {}  ".format(_text((result.get("regime") or {}).get("breadth_sample_size")), (result.get("regime") or {}).get("breadth_reliable")),
    ])
    lines.extend("- {}".format(reason) for reason in (result.get("regime") or {}).get("reasoning") or [])
    lines.extend(["", "## Sector rotation", "", "| Rank | ETF | State | 5d RS | 20d RS | 60d RS |", "|---:|---|---|---:|---:|---:|"])
    sectors = sorted((result.get("sector_rotation") or {}).values(), key=lambda row: int(row.get("rank") or 999))
    for row in sectors[:15]:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(row.get("rank"), row.get("ticker"), row.get("state"), _pct(row.get("rs_5d")), _pct(row.get("rs_20d")), _pct(row.get("rs_60d"))))
    lines.extend(["", "## Sourced market/event calendar", ""])
    market_events = (result.get("context") or {}).get("market_events") or []
    if market_events:
        lines.extend("- {} — {} — {} — {}".format(_text(row.get("event_date")), _text(row.get("title") or row.get("claim")), _text(row.get("source")), _text(row.get("source_url"))) for row in market_events)
    else:
        lines.append("- {} — no sourced macro/market event calendar was supplied.".format(DATA_UNAVAILABLE))
    lines.extend(
        [
            "",
            "## Scan funnel",
            "",
            "| Stage | Count |",
            "|---|---:|",
            "| Configured seed rows | {} |".format(diagnostics.get("configured_universe_rows", 0)),
            "| ORATS discovery rows selected | {} |".format(diagnostics.get("discovery_selected_rows", 0)),
            "| Initially scanned | {} |".format(diagnostics.get("initially_scanned", 0)),
            "| Missing sufficient technical history | {} |".format(diagnostics.get("missing_technical_history", 0)),
            "| Passing underlying liquidity | {} |".format(diagnostics.get("passing_liquidity", 0)),
            "| Passing technical setup | {} |".format(diagnostics.get("passing_technical_setup", 0)),
            "| Triggered setups | {} |".format(diagnostics.get("triggered_setups", 0)),
            "| Triggered names with historical paths | {} |".format(diagnostics.get("history_with_samples_for_triggered", 0)),
            "| Triggered names with option chain requested | {} |".format(diagnostics.get("option_chains_requested", 0)),
            "| Requested chains returning rows | {} |".format(diagnostics.get("option_chains_with_rows", 0)),
            "| Exact option structures evaluated on training paths | {} |".format(diagnostics.get("option_structures_evaluated", 0)),
            "| With aligned sourced catalyst/regime evidence | {} |".format(diagnostics.get("passing_catalyst_regime", 0)),
            "| With preferred target asymmetry | {} |".format(diagnostics.get("passing_risk_reward", 0)),
            "| Option alternatives withheld for earnings timing | {} |".format(diagnostics.get("rejected_for_earnings", 0)),
            "| Option structures rejected for pricing/liquidity | {} |".format(diagnostics.get("rejected_for_options", 0)),
            "| Positive held-out option alternatives before vehicle/earnings choice | {} |".format(diagnostics.get("positive_option_alternatives", 0)),
            "| Positive-EV option target trades | {} |".format(diagnostics.get("option_target_trades", 0)),
            "| Positive-EV stock target trades | {} |".format(diagnostics.get("stock_target_trades", 0)),
            "| Positive-EV target trades | {} |".format(diagnostics.get("target_trades", diagnostics.get("finally_qualifying", 0))),
            "",
            "## Final ranking",
            "",
            "| Rank | Ticker | Direction | Vehicle | Setup | Entry | Stop | Target | Modeled POP | Expected profit | Horizon | Earnings Risk | Status |",
            "|---:|---|---|---|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for rank, candidate in enumerate(candidates, start=1):
        lines.append(_idea_row(rank, candidate))
    for status in (TARGET_TRADE, SETUP_ONLY, NO_POSITIVE_EDGE, WATCHLIST, REJECTED):
        matching = [item for item in candidates if item.get("status") == status]
        lines.extend(["", "### {}".format(status), ""])
        if matching:
            lines.extend("- {} — {} — score {} — {}".format(item.get("ticker"), (item.get("setup") or {}).get("name"), item.get("score"), "; ".join(_all_reasons(item) or ["no blocker recorded"])) for item in matching)
        else:
            lines.append("- None.")
    if len(targets) < 3:
        lines.extend(["", "## Five strongest non-trades", ""])
        near = [item for item in candidates if item.get("status") != TARGET_TRADE][:5]
        lines.extend("- **{}** — {} — reason: {}".format(item.get("ticker"), (item.get("setup") or {}).get("name"), "; ".join(_all_reasons(item) or ["entry trigger is absent"])) for item in near)
    lines.extend(["", "## Detailed research cards", ""])
    for candidate in candidates:
        lines.append(render_candidate(candidate))
        lines.append("")
    lines.extend(
        [
            "## ORATS and run sources",
            "",
            "| Source | Endpoint | Status | Rows | Latest data | Fetched | Cache SHA-256 |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for trace in result.get("source_traces") or []:
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(trace.get("source"), trace.get("endpoint"), trace.get("status"), trace.get("rows"), _table_escape(trace.get("latest_data_at")), _table_escape(trace.get("fetched_at_utc")), str(trace.get("cache_sha256") or "")[:16]))
    lines.extend(
        [
            "",
            "## Missing-data contract",
            "",
            "Missing source families are shown as **DATA UNAVAILABLE**. CORAT does not infer ORATS values, X posts, catalysts, live fills, or dealer positioning. Modeled POP is labeled as CORAT output and calculated only from displayed same-setup historical samples; it is not mislabeled as ORATS POP or as a guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def render_board_csv(candidates: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "rank", "ticker", "direction", "vehicle", "setup", "status", "score", "confidence",
        "price", "entry_low", "entry_high", "stop", "target_1", "target_2", "reward_risk_2",
        "modeled_pop", "model_sample_size", "expected_profit", "profit_factor", "option_strategy", "option_expiration",
        "option_legs", "option_entry_side", "option_limit", "maximum_loss", "next_earnings_date", "next_earnings_reference",
        "blockers", "review_notes", "hard_rejections",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for rank, candidate in enumerate(candidates, start=1):
        technical = candidate.get("technical") or {}
        plan = candidate.get("stock_plan") or {}
        setup = candidate.get("setup") or {}
        vol = candidate.get("volatility") or {}
        economics = candidate.get("economics") or {}
        option = candidate.get("option") or {}
        expected_profit = economics.get("expected_profit_dollars") if candidate.get("vehicle") == "OPTIONS" else economics.get("expected_profit_per_share")
        writer.writerow(
            {
                "rank": rank,
                "ticker": candidate.get("ticker"),
                "direction": setup.get("direction"),
                "vehicle": candidate.get("vehicle"),
                "setup": setup.get("name"),
                "status": candidate.get("status"),
                "score": candidate.get("score"),
                "confidence": candidate.get("confidence"),
                "price": technical.get("price"),
                "entry_low": plan.get("entry_low"),
                "entry_high": plan.get("entry_high"),
                "stop": plan.get("stop"),
                "target_1": plan.get("target_1"),
                "target_2": plan.get("target_2"),
                "reward_risk_2": plan.get("reward_risk_2"),
                "modeled_pop": economics.get("modeled_pop"),
                "model_sample_size": economics.get("model_sample_size"),
                "expected_profit": expected_profit,
                "profit_factor": economics.get("profit_factor"),
                "option_strategy": option.get("strategy"),
                "option_expiration": option.get("expiration"),
                "option_legs": _option_legs(candidate),
                "option_entry_side": option.get("debit_credit"),
                "option_limit": option.get("expected_entry"),
                "maximum_loss": option.get("maximum_loss") if candidate.get("vehicle") == "OPTIONS" else plan.get("risk_per_share"),
                "next_earnings_date": vol.get("next_earnings_date"),
                "next_earnings_reference": vol.get("next_earnings_reference"),
                "blockers": "; ".join(candidate.get("blockers") or []),
                "review_notes": "; ".join(candidate.get("review_notes") or []),
                "hard_rejections": "; ".join(candidate.get("hard_rejections") or []),
            }
        )
    return stream.getvalue()
