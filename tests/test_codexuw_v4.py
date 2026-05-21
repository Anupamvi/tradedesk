from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from codexuw.daily_v4 import (
    PIPELINE_NAME_V4,
    _default_out_dir,
    _disposition,
    _hard_blocker_reason,
    apply_v4_risk_cap,
    apply_v4_safety_calibration,
    build_no_miss_audit,
    build_construction_attempts,
    build_candidate_disposition,
    build_secondary_liquidity_sweep,
    build_suppression_audit,
    build_v4_safety_calibration,
    build_v4_swing_target_tickets,
    build_v4_target_model,
    parse_args,
    run_v4_daily,
    write_v4_outputs,
)


ASOF = dt.date(2026, 5, 20)
EXPIRY = "2026-05-29"


def _candidate(**overrides) -> dict:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "expiry": EXPIRY,
        "dte": 9,
        "trade_status": "Research",
        "trade_tier": "",
        "trade_status_reason": "credit target miss but thesis remains reviewable",
        "hard_rejects": "",
        "penalties": "credit_below_min_16pct_width",
        "credit": 0.75,
        "mid_credit": 0.75,
        "natural_credit": 0.62,
        "required_entry": 0.90,
        "credit_pct_width": 0.15,
        "spread_width": 5.0,
        "max_profit": 75.0,
        "max_loss": 425.0,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "short_leg": "AAA260529P00100000",
        "long_leg": "AAA260529P00095000",
        "live_status": "PASS",
        "quote_width_pct": 0.08,
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "oi_carryover_reason": "exact-leg OI supports direction",
        "edge_verdict": "acceptable",
        "replay_ev_verdict": "acceptable",
        "edge_sample_size": 12,
        "edge_win_rate": 0.62,
        "edge_avg_pnl": 24.0,
        "confirmation_score": 7.2,
        "score": 6.5,
        "target_entry": 0.90,
        "price_annotation": "current credit $0.75 is below target $0.90; show as work-limit",
    }
    row.update(overrides)
    return row


def test_v4_cli_and_default_output_folders_say_v4(tmp_path: Path) -> None:
    args = parse_args(["run", "--date", "2026-05-20"])

    assert args.command == "run"
    assert args.report_mode == "post-close"
    assert args.index_income_mode == "primary"
    assert args.risk_budget == 0
    assert _default_out_dir(tmp_path, ASOF, "run") == tmp_path / "out" / "codexdaily_v4_2026-05-20"
    assert _default_out_dir(tmp_path, ASOF, "validation") == tmp_path / "out" / "codexdaily_v4_validation_2026-05-20"
    assert _default_out_dir(tmp_path, ASOF, "overlay", dt.date(2026, 5, 21)) == tmp_path / "out" / "codexdaily_v4_overlay_2026-05-20_overlay_2026-05-21"


def test_price_target_miss_stays_visible_as_v4_work_limit_ticket() -> None:
    scored = pd.DataFrame([_candidate()])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=top_flow,
    )

    assert len(tickets) == 1
    ticket = tickets.iloc[0]
    assert ticket["final disposition"] == "Swing Target / Work Limit"
    assert ticket["next-session swing entry target"] == ">= $0.90 credit"
    assert "sell AAA 2026-05-29 100P / buy AAA 2026-05-29 95P" in ticket["trade legs"]
    assert "AAA260529" not in ticket["trade legs"]
    assert "credit is 15.0% of $5 width" in ticket["target price methodology"]


def test_v4_hard_event_risk_blocks_target_ticket() -> None:
    row = _candidate(catalyst_earnings_days=0, catalyst_status="caution", penalties="earnings_news_risk")
    scored = pd.DataFrame([row])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame(),
    )

    assert tickets.empty
    assert _disposition(pd.Series(row), targetable=True) == "Avoid"
    assert _hard_blocker_reason(pd.Series(row)) == "earnings/event risk invalidates the structure"


def test_no_miss_audit_disposes_top_flow_tickers_without_candidates() -> None:
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"},
            {"rank": 2, "ticker": "BBB", "net_premium": 1_500_000, "flow_direction": "bearish"},
        ]
    )
    scored = pd.DataFrame([_candidate(ticker="AAA")])
    tickets = build_v4_swing_target_tickets(scored=scored, board=pd.DataFrame(), regime={"trend": "uptrend"}, top_flow=top_flow)
    dispositions = build_candidate_disposition(candidates=scored, scored=scored, top_flow=top_flow, tickets=tickets)
    attempts = build_construction_attempts(scored=scored, top_flow=top_flow, tickets=tickets, portfolio={"status": "ok"})

    audit = build_no_miss_audit(top_flow=top_flow, scored=scored, dispositions=dispositions, attempts=attempts, tickets=tickets)
    missing = audit[audit["ticker"].eq("BBB")].iloc[0]

    assert bool(missing["candidate_generated"]) is False
    assert missing["constructions_attempted"] == 0
    assert missing["final_disposition"] == "Research"
    assert "no candidate generated" in missing["if_not_targetable_exact_reason"]


def test_suppression_audit_flags_no_price_miss_silent_drop() -> None:
    scored = pd.DataFrame([_candidate(), _candidate(ticker="CCC", hard_rejects="no_usable_liquidity")])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA"}, {"rank": 2, "ticker": "CCC"}])
    tickets = build_v4_swing_target_tickets(scored=scored.iloc[[0]], board=pd.DataFrame(), regime={"trend": "uptrend"}, top_flow=top_flow)
    dispositions = build_candidate_disposition(candidates=scored, scored=scored, top_flow=top_flow, tickets=tickets)

    suppression = build_suppression_audit(dispositions)

    assert not suppression.empty
    assert not suppression["targetable_trade_hidden_by_price_miss"].any()
    assert "CCC" in set(suppression["ticker"])


def test_v4_target_model_uses_swing_tickets_not_only_execute() -> None:
    tickets = pd.DataFrame(
        [
            {"profit target": "$500.00", "max loss": "$1,000.00", "final disposition": "Swing Target / Work Limit", "setup family": "Credit spreads", "expected win rate": "60%"},
            {"profit target": "$300.00", "max loss": "$500.00", "final disposition": "Scout", "setup family": "Debit spreads", "expected win rate": "50%"},
        ]
    )

    model = build_v4_target_model(
        asof=ASOF,
        tickets=tickets,
        portfolio={"status": "ok", "cash": 10_000},
        monthly_profit_target=10_000,
        month_to_date_realized_pnl=1_000,
        open_unrealized_pnl=0,
        risk_budget=5_000,
    )

    assert model["execute_profit_potential"] == 0
    assert model["swing_target_profit_potential_if_filled"] == 800
    assert model["realistic_fill_adjusted_target_potential"] > 0
    assert model["required_number_of_target_tickets"] is not None


def test_v4_strategy_slump_mutes_exact_family_to_research() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute")])
    ledger = pd.DataFrame(
        [
            {"report_date": "2026-05-18", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -100},
            {"report_date": "2026-05-17", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -50},
            {"report_date": "2026-05-16", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
        ]
    )

    calibration = build_v4_safety_calibration(scored=scored, outcome_ledger=ledger, asof=ASOF)
    adjusted = apply_v4_safety_calibration(scored, calibration)

    assert bool(calibration.iloc[0]["strategy_slump_muted"]) is True
    assert adjusted.iloc[0]["trade_status"] == "Research"
    assert "v4_strategy_slump_muted" in adjusted.iloc[0]["penalties"]
    assert _disposition(adjusted.iloc[0], targetable=True) == "Research"


def test_v4_negative_shadow_ev_downgrades_execute_without_slump() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute")])
    ledger = pd.DataFrame(
        [
            {"report_date": "2026-05-18", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -200},
            {"report_date": "2026-05-17", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -100},
            {"report_date": "2026-05-16", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
            {"report_date": "2026-05-15", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
            {"report_date": "2026-05-14", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
        ]
    )

    calibration = build_v4_safety_calibration(scored=scored, outcome_ledger=ledger, asof=ASOF)
    adjusted = apply_v4_safety_calibration(scored, calibration)

    assert calibration.iloc[0]["shadow_backtest_status"] == "negative_ev"
    assert bool(calibration.iloc[0]["strategy_slump_muted"]) is False
    assert adjusted.iloc[0]["trade_status"] == "Research"
    assert "v4_negative_shadow_ev" in adjusted.iloc[0]["penalties"]


def test_v4_target_ticket_includes_gap_risk_and_oco_bracket_for_execute() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute", penalties="", trade_status_reason="all live checks pass")])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}]),
    )

    assert tickets.iloc[0]["final disposition"] == "Execute"
    assert "Gap +1%" in tickets.iloc[0]["gap-risk plan +/-1% open"]
    assert "OCO" in tickets.iloc[0]["OCO bracket order logic"]
    assert "BUY TO CLOSE" in tickets.iloc[0]["OCO bracket order logic"]


def test_v4_hard_risk_cap_truncates_or_downgrades_target_signals() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade legs": "spread",
                "final disposition": "Swing Target / Work Limit",
                "max loss": "$10,000.00",
                "profit target": "$1,000.00",
                "suggested size": "5 contracts; only if checks pass",
                "blocker before entry": "",
                "manual review instruction": "",
                "safety calibration flags": "",
            },
            {
                "ticker": "BBB",
                "trade legs": "spread",
                "final disposition": "Execute",
                "max loss": "$3,000.00",
                "profit target": "$300.00",
                "suggested size": "1 contract; only if checks pass",
                "blocker before entry": "",
                "manual review instruction": "",
                "safety calibration flags": "",
            },
        ]
    )

    capped, audit = apply_v4_risk_cap(tickets, {"status": "ok", "total_value": 100_000})

    assert capped.set_index("ticker").loc["AAA", "max loss"] == "$2,000.00"
    assert "Risk Capped" in capped.set_index("ticker").loc["AAA", "safety calibration flags"]
    assert capped.set_index("ticker").loc["BBB", "final disposition"] == "Research"
    assert int(audit["risk_capped"].sum()) == 2


def test_v4_secondary_liquidity_sweep_triggers_below_three_candidates() -> None:
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 1_000_000, "volume_oi_ratio": 0.7, "max_rolling_5m_premium": 800_000},
            {"rank": 2, "ticker": "BBB", "net_premium": 900_000, "volume_oi_ratio": 0.2, "max_rolling_5m_premium": 500_000},
        ]
    )
    velocity = pd.DataFrame([{"ticker": "AAA", "rolling_5m_premium": 800_000, "rolling_15m_premium": 1_500_000, "flow_velocity_signal": True}])
    correlation = pd.DataFrame([{"ticker": "AAA", "benchmark": "SPY", "rolling_correlation": 0.91, "reason": "beta noise"}])

    sweep = build_secondary_liquidity_sweep(
        candidates=pd.DataFrame([_candidate()]),
        scored=pd.DataFrame([_candidate()]),
        top_flow=top_flow,
        flow_velocity=velocity,
        correlation=correlation,
    )

    assert sweep["triggered"].all()
    first = sweep.set_index("ticker").loc["AAA"]
    assert bool(first["relaxed_uw_block_size_filters"]) is True
    assert first["flow_velocity_scan"] == "pass"
    assert bool(first["beta_noise_ignored"]) is True


def test_write_v4_outputs_writes_v4_report_order_and_required_artifacts_without_v3_core(tmp_path: Path) -> None:
    base_dir = tmp_path / "2026-05-20"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    scored = pd.DataFrame([_candidate()])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}])
    args = parse_args(["run", "--date", "2026-05-20", "--out-dir", str(out_dir)])

    manifest = write_v4_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        asof=ASOF,
        args=args,
        candidates=scored,
        scored=scored,
        board=pd.DataFrame(),
        top_flow=top_flow,
        flow_velocity=pd.DataFrame(),
        correlation=pd.DataFrame(),
        macro=pd.DataFrame(),
        confirmation=pd.DataFrame(),
        data_quality={"status": "ok", "items": []},
        portfolio={"status": "ok", "cash": 25_000, "risk_actions": []},
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        regime_context={"base_regime": {"trend": "uptrend", "volatility": "low", "flow": "weak"}},
        recent_performance={"status": "unavailable"},
        live_outcomes={"status": "unavailable"},
        loss_review={"status": "unavailable"},
        liquidity_summary={"status": "ok"},
    )

    report = Path(manifest["report_path"]).read_text(encoding="utf-8")
    assert manifest["pipeline_name"] == PIPELINE_NAME_V4
    assert "| Pipeline | Codex Daily V4 |" in report
    ordered = [
        "## First Screen",
        "## Market Insight For Tomorrow",
        "## Swing Target Tickets For Tomorrow",
        "## Portfolio Repair / Open Risk",
        "## $10k/month Target Math",
        "## No-Miss Audit",
        "## Opportunity Board",
        "## Detailed artifacts",
    ]
    positions = [report.index(item) for item in ordered]
    assert positions == sorted(positions)
    for name in [
        "codexdaily_v4_raw_universe_2026-05-20.csv",
        "codexdaily_v4_candidate_disposition_2026-05-20.csv",
        "codexdaily_v4_swing_target_tickets_2026-05-20.csv",
        "codexdaily_v4_suppression_audit_2026-05-20.csv",
        "codexdaily_v4_construction_attempts_2026-05-20.csv",
        "codexdaily_v4_no_miss_audit_2026-05-20.csv",
        "codexdaily_v4_safety_calibration_2026-05-20.csv",
        "codexdaily_v4_risk_cap_audit_2026-05-20.csv",
        "codexdaily_v4_secondary_liquidity_sweep_2026-05-20.csv",
    ]:
        assert (out_dir / name).exists()


def test_v4_max_final_trades_is_not_a_visibility_cap(tmp_path: Path) -> None:
    base_dir = tmp_path / "2026-05-20"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    scored = pd.DataFrame(
        [
            _candidate(ticker="AAA", trade_status="Execute", penalties="", credit=1.0, mid_credit=1.0, required_entry=0.9, target_entry=0.9),
            _candidate(ticker="BBB", trade_status="Execute", penalties="", credit=1.1, mid_credit=1.1, required_entry=0.9, target_entry=0.9),
        ]
    )
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"},
            {"rank": 2, "ticker": "BBB", "net_premium": 1_500_000, "flow_direction": "bullish"},
        ]
    )
    args = parse_args(["run", "--date", "2026-05-20", "--out-dir", str(out_dir), "--max-final-trades", "1"])

    manifest = write_v4_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        asof=ASOF,
        args=args,
        candidates=scored,
        scored=scored,
        board=pd.DataFrame(),
        top_flow=top_flow,
        flow_velocity=pd.DataFrame(),
        correlation=pd.DataFrame(),
        macro=pd.DataFrame(),
        confirmation=pd.DataFrame(),
        data_quality={"status": "ok", "items": []},
        portfolio={"status": "ok", "cash": 25_000, "total_value": 100_000, "risk_actions": []},
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        regime_context={"base_regime": {"trend": "uptrend", "volatility": "low", "flow": "weak"}},
        recent_performance={"status": "unavailable"},
        live_outcomes={"status": "unavailable"},
        loss_review={"status": "unavailable"},
        liquidity_summary={"status": "ok"},
    )

    assert manifest["opportunity_counts"]["execute"] == 2
    assert manifest["visible_signal_policy"]["active_execute_cap"] is None
    assert manifest["visible_signal_policy"]["max_final_trades_arg"] == 1
    assert manifest["visible_signal_policy"]["aggregate_risk_budget_applied"] is False
    assert manifest["target_model"]["aggregate_risk_budget_applied"] is False
    report = Path(manifest["report_path"]).read_text(encoding="utf-8")
    assert "Visible signal cap | none" in report
    assert "Aggregate risk budget | not configured" in report


def test_run_v4_daily_is_not_a_v3_wrapper() -> None:
    names = run_v4_daily.__code__.co_names

    assert "run_v3_daily" not in names
    assert "write_v4_outputs_from_core" not in names
