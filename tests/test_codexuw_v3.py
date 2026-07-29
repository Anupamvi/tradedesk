from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from codexuw.confirmations import apply_confirmation_evidence, build_confirmation_evidence
from codexuw.daily_v3 import parse_args, write_v3_data_error_report
from codexuw.engine import assign_trade_statuses, select_final_trades
from codexuw.fallback_income import apply_fallback_income_status, build_fallback_income_candidates
from codexuw.leg_drift import build_leg_drift_audit
from codexuw.lifecycle import apply_lifecycle_triggers
from codexuw.liquidity_shift import (
    apply_liquidity_shift_context,
    build_correlation_anomalies,
    build_liquidity_shift_signals,
    build_zero_dte_gamma_context,
    compute_volatility_thresholds,
)
from codexuw.loss_review import apply_loss_review, review_recent_losses
from codexuw.macro_gates import build_macro_event_gates
from codexuw.missed_opportunity import build_missed_opportunity_audit
from codexuw.opportunity import build_opportunity_board, build_target_ticket_board, classify_no_trade_audit, opportunity_counts, write_recommendation_ledger
from codexuw.overlay import run_overlay
from codexuw.snapshots import summarize_schwab_chain_snapshots
from codexuw.target_model import build_v3_target_model
from codexuw.validation import select_systematic_date_folders


ASOF = dt.date(2026, 5, 19)
EXPIRY = dt.date(2026, 5, 29)


def _candidate(**overrides) -> dict:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "expiry": EXPIRY,
        "dte": 10,
        "trade_status": "Execute",
        "trade_tier": "Execute A",
        "trade_status_reason": "live pricing, liquidity, edge, and risk gates passed",
        "hard_rejects": "",
        "penalties": "",
        "credit": 1.2,
        "mid_credit": 1.2,
        "natural_credit": 1.05,
        "required_entry": 1.2,
        "credit_pct_width": 0.24,
        "spread_width": 5.0,
        "max_profit": 120.0,
        "max_loss": 380.0,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "short_delta": -0.22,
        "short_leg": "AAA260529P00100000",
        "long_leg": "AAA260529P00095000",
        "live_status": "PASS",
        "quote_width_pct": 0.08,
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "replay_ev_verdict": "positive",
        "edge_verdict": "positive",
        "edge_sample_size": 14,
        "edge_win_rate": 0.64,
        "edge_avg_pnl": 52.0,
        "confirmation_score": 8.5,
        "score": 8.0,
        "confidence": "High",
    }
    row.update(overrides)
    return row


def test_v3_data_error_report_manifest_and_report_say_v3(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-19"
    out_dir = tmp_path / "out"
    base_dir.mkdir()

    manifest = write_v3_data_error_report(out_dir, ASOF, base_dir, FileNotFoundError("missing UW export"))

    report = Path(manifest["report_path"]).read_text(encoding="utf-8")
    data = json.loads((out_dir / "codexdaily_v3_manifest_2026-05-19.json").read_text(encoding="utf-8"))
    assert data["pipeline_name"] == "Codex Daily V3"
    assert data["pipeline_version"] == "v3.2-profit-integrity-20260719"
    assert "| Pipeline | Codex Daily V3 |" in report
    assert "| Version lock | locked 2026-07-19; supersedes v3.0, v3.1-exec-confidence-20260612-143405 |" in report
    assert "Codex Daily V2" not in report


def test_v3_cli_exposes_required_modes() -> None:
    assert parse_args(["run", "--date", "2026-05-19"]).command == "run"
    assert parse_args(["intraday", "--date", "2026-05-19"]).command == "intraday"
    assert parse_args(["validate", "--as-of", "2026-05-19", "--latest-n", "3"]).latest_n == 3
    assert parse_args(["loss-review", "--as-of", "2026-05-19"]).command == "loss-review"
    assert parse_args(["overlay", "--date", "2026-05-19", "--overlay-file", "chain-oi-changes-2026-05-20.zip"]).command == "overlay"


def test_no_execute_still_creates_opportunity_board_with_lane_alternatives() -> None:
    scored = pd.DataFrame(
        [
            _candidate(trade_status="Research", trade_status_reason="news_unconfirmed", penalties="news_unconfirmed"),
            _candidate(
                ticker="BBB",
                direction="Bull Call",
                strategy="Bull Call Debit Spread",
                trade_status="Research",
                debit=2.0,
                mid_debit=2.0,
                natural_debit=2.1,
                max_profit=300.0,
                max_loss=200.0,
                short_leg="BBB260529C00105000",
                long_leg="BBB260529C00100000",
            ),
            _candidate(ticker="SPY", index_fallback=True, trade_status="Research"),
        ]
    )

    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 25_000, "risk_actions": []})
    lanes = set(board["Lane"])

    assert "Momentum Debit" in lanes
    assert "Index/ETF" in lanes
    assert "Portfolio Repair" in lanes
    assert "Wheel/Cash" in lanes
    assert opportunity_counts(board)["execute"] == 0


def test_scout_cannot_become_execute_in_v3_board() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Watch", trade_tier="manual-confirmation-scout", trade_status_reason="news_unconfirmed")])
    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    counts = opportunity_counts(board)
    assert counts["scout"] >= 1
    assert counts["execute"] == 0
    assert "Scout" in board.iloc[0]["Status"]


def test_credit_target_miss_becomes_visible_work_limit_not_research() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                credit=0.55,
                mid_credit=0.55,
                natural_credit=0.45,
                credit_pct_width=0.11,
                spread_width=5.0,
                confirmation_score=6.5,
                replay_ev_verdict="acceptable",
                edge_verdict="acceptable",
            )
        ]
    )

    out = assign_trade_statuses(scored)
    board = build_opportunity_board(scored=out, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    assert out.iloc[0]["trade_status"] == "Watch"
    assert out.iloc[0]["trade_tier"] == "work-limit-price-target"
    assert "credit must improve" in out.iloc[0]["what_must_improve"]
    assert "Work Limit" in board.iloc[0]["Status"]


def test_negative_edge_credit_cannot_become_v3_work_limit() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                ticker="NOW",
                credit=1.67,
                mid_credit=1.67,
                natural_credit=1.60,
                required_entry=1.40,
                confirmation_score=9.0,
                replay_ev_verdict="acceptable_secondary_income",
                edge_verdict="acceptable_secondary_income",
                edge_sample_size=17,
                edge_win_rate=0.5294117647,
                edge_avg_pnl=-33.897,
            )
        ]
    )

    out = assign_trade_statuses(scored)
    board = build_opportunity_board(scored=out, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})
    targets = build_target_ticket_board(board)
    now = board[board["Ticker"].eq("NOW")].iloc[0]

    assert out.iloc[0]["trade_status"] in {"Research", "Avoid"}
    assert "negative_edge_avg_pnl" in out.iloc[0]["trade_status_reason"]
    assert "Work Limit" not in now["Status"]
    assert "Scout" not in now["Status"]
    assert "Execute" not in now["Status"]
    assert targets.empty or "NOW" not in set(targets["Ticker"])


def test_fallback_income_cannot_overwrite_negative_edge() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                ticker="NOW",
                construction_source="fallback_income",
                live_construction_source="fallback_income",
                fallback_target_credit=1.40,
                target_entry=1.40,
                credit=1.67,
                mid_credit=1.67,
                natural_credit=1.60,
                quote_width_pct=0.05,
                edge_sample_size=17,
                edge_win_rate=0.5294117647,
                edge_avg_pnl=-33.897,
                replay_ev_verdict="thin_sample",
                edge_verdict="thin_sample",
            )
        ]
    )

    out = apply_fallback_income_status(scored)

    assert out.iloc[0]["trade_status"] == "Avoid"
    assert out.iloc[0]["trade_tier"] == "fallback-income-weak-edge"
    assert out.iloc[0]["edge_verdict"] == "negative"
    assert "acceptable_secondary_income cannot override" in out.iloc[0]["trade_status_reason"]


def test_leg_drift_audit_flags_changed_short_strike_and_width() -> None:
    recommendations = pd.DataFrame(
        [
            {
                "ticker": "NOW",
                "generated_at": "2026-05-30T19:03:35Z",
                "trade": "Bull Put Credit Spread: sell NOW 2026-07-17 115P / buy NOW 2026-07-17 110P",
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "ticker": "NOW",
                "trade": "Bull Put Credit Spread: sell NOW 2026-07-17 120P / buy NOW 2026-07-17 110P",
            }
        ]
    )

    audit = build_leg_drift_audit(recommendations, fills)

    assert bool(audit.iloc[0]["drift_detected"]) is True
    assert audit.iloc[0]["status"] == "UNAPPROVED LEG DRIFT - re-score required"
    assert "sell_strike_changed:115->120" in audit.iloc[0]["drift_reason"]
    assert "width_changed:5->10" in audit.iloc[0]["drift_reason"]


def test_debit_target_miss_becomes_visible_work_limit_not_research() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                direction="Bull Call",
                strategy="Bull Call Debit Spread",
                credit=math.nan,
                credit_pct_width=math.nan,
                debit=3.40,
                mid_debit=3.40,
                natural_debit=3.65,
                debit_pct_width=0.68,
                spread_width=5.0,
                reward_risk=0.47,
                confirmation_score=6.5,
                replay_ev_verdict="acceptable",
                edge_verdict="acceptable",
                short_leg="AAA260529C00105000",
                long_leg="AAA260529C00100000",
            )
        ]
    )

    out = assign_trade_statuses(scored)
    board = build_opportunity_board(scored=out, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    assert out.iloc[0]["trade_status"] == "Watch"
    assert out.iloc[0]["trade_tier"] == "work-limit-price-target"
    assert "debit must fall" in out.iloc[0]["what_must_improve"]
    assert "Work Limit" in board.iloc[0]["Status"]


def test_eod_swing_target_board_shows_targets_not_just_execute() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Watch",
                trade_tier="work-limit-price-target",
                trade_status_reason="work target only",
                credit=0.80,
                required_entry=0.90,
                flow_quality="directional",
                oi_carryover_status="supportive",
                edge_verdict="acceptable",
                confirmation_score=6.5,
            ),
            _candidate(
                ticker="BBB",
                trade_status="Research",
                trade_status_reason="thin sample but targetable",
                credit=1.10,
                required_entry=0.90,
                flow_quality="directional",
                oi_carryover_status="supportive",
                edge_verdict="positive",
                confirmation_score=5.5,
            ),
        ]
    )
    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    targets = build_target_ticket_board(board)

    assert len(targets) >= 2
    assert "Next-session swing entry" in targets.columns
    assert targets["Swing trend evidence"].astype(str).str.contains("flow=directional").any()
    assert targets["Swing work instruction"].astype(str).str.contains("Work the limit|Target only", regex=True).any()


def test_v3_board_and_targets_are_uncapped_by_default() -> None:
    scored = pd.DataFrame(
        [
            _candidate(ticker=f"T{i:02d}", max_profit=100.0 + i, target_profit_total=60.0 + i)
            for i in range(16)
        ]
    )

    board = build_opportunity_board(
        scored=scored,
        final=scored,
        watchlist=pd.DataFrame(),
        portfolio={"status": "ok", "cash": 25_000},
    )
    targets = build_target_ticket_board(board)

    execute_rows = board[board["Status"].astype(str).str.contains("Execute", regex=False)]
    assert len(execute_rows) == 16
    assert set(execute_rows["Ticker"]) <= set(targets["Ticker"])
    assert len(targets) >= 16


def test_v3_board_formats_trade_legs_as_human_order_legs() -> None:
    credit = pd.DataFrame([_candidate()])
    credit_board = build_opportunity_board(scored=credit, final=credit, watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    assert "sell AAA 2026-05-29 100P / buy AAA 2026-05-29 95P" in credit_board.iloc[0]["Trade"]
    assert "AAA260529P00100000" not in credit_board.iloc[0]["Trade"]

    debit = pd.DataFrame(
        [
            _candidate(
                direction="Bull Call",
                strategy="Bull Call Debit Spread",
                short_leg="BBB260529C00105000",
                long_leg="BBB260529C00100000",
                debit=2.0,
                mid_debit=2.0,
                natural_debit=2.1,
            )
        ]
    )
    debit_board = build_opportunity_board(scored=debit, final=debit, watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 10_000})

    assert "buy BBB 2026-05-29 100C / sell BBB 2026-05-29 105C" in debit_board.iloc[0]["Trade"]
    assert "BBB260529C00105000" not in debit_board.iloc[0]["Trade"]


def test_confirmation_evidence_can_clear_manual_scout_blockers() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Watch",
                penalties="news_unconfirmed;flow_not_directional:unclear;portfolio_warning",
                flow_quality="unclear",
                vwap_confirmation="bullish_above_tape_vwap",
                flow_velocity_signal=True,
                catalyst_status="unknown",
                next_earnings_dt="2026-06-15",
            )
        ]
    )

    evidence = build_confirmation_evidence(
        scored=scored,
        asof=ASOF,
        input_provenance={"browser_text_count": 2},
    )
    out = apply_confirmation_evidence(scored, evidence)

    assert evidence.iloc[0]["confirmation_status"] == "cleared"
    assert "news_unconfirmed" not in out.iloc[0]["penalties"]
    assert "flow_not_directional:unclear" not in out.iloc[0]["penalties"]
    assert "portfolio_warning" in out.iloc[0]["penalties"]
    assert out.iloc[0]["flow_quality"] == "directional"
    assert out.iloc[0]["catalyst_status"] == "mixed"


def test_confirmation_evidence_keeps_hard_event_blocker() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Watch",
                penalties="news_unconfirmed;flow_not_directional:unclear",
                flow_quality="directional",
                catalyst_status="unknown",
                next_earnings_dt="2026-05-22",
            )
        ]
    )

    evidence = build_confirmation_evidence(scored=scored, asof=ASOF, input_provenance={"browser_text_count": 1})
    out = apply_confirmation_evidence(scored, evidence)

    assert evidence.iloc[0]["confirmation_status"] == "blocked"
    assert "news_unconfirmed" in out.iloc[0]["penalties"]


def test_confirmation_evidence_removes_only_cleared_blocker_family() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Watch",
                penalties="news_unconfirmed;flow_not_directional:unclear",
                flow_quality="unclear",
                vwap_confirmation="bearish_vwap_not_confirmed",
                catalyst_status="unknown",
                next_earnings_dt="2026-06-15",
            )
        ]
    )

    evidence = build_confirmation_evidence(scored=scored, asof=ASOF, input_provenance={"browser_text_count": 1})
    out = apply_confirmation_evidence(scored, evidence)

    assert evidence.iloc[0]["confirmation_status"] == "manual"
    assert "news_unconfirmed" not in out.iloc[0]["penalties"]
    assert "flow_not_directional:unclear" in out.iloc[0]["penalties"]
    assert out.iloc[0]["flow_quality"] == "unclear"


def test_confirmation_evidence_does_not_require_company_earnings_news_for_etf() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                ticker="QQQ",
                trade_status="Watch",
                penalties="news_unconfirmed",
                flow_quality="directional",
                catalyst_status="unknown",
                next_earnings_dt=None,
            )
        ]
    )

    evidence = build_confirmation_evidence(
        scored=scored,
        asof=ASOF,
        input_provenance={"browser_text_count": 0},
    )
    out = apply_confirmation_evidence(scored, evidence)

    assert evidence.iloc[0]["news_confirmation"] == "cleared"
    assert evidence.iloc[0]["confirmation_status"] == "cleared"
    assert "news_unconfirmed" not in out.iloc[0]["penalties"]
    assert out.iloc[0]["catalyst_status"] == "mixed"


def test_wheel_cash_lane_uses_priced_cash_secured_put_when_available() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Research",
                sell_leg_bid=1.1,
                sell_leg_ask=1.3,
                sell_leg_mid=1.2,
                quote_width_pct=0.08,
                short_oi=900,
                short_volume=120,
            )
        ]
    )

    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 25_000})
    wheel = board[board["Lane"].eq("Wheel/Cash")].iloc[0]

    assert wheel["Ticker"] == "AAA"
    assert "Cash-secured put (assignment-risk): sell AAA 2026-05-29 100P" == wheel["Trade"]
    assert "requires live CSP chain pricing" not in wheel["Entry limit"]
    assert wheel["Max loss"] == "$9,880.00"


def test_wheel_cash_lane_deduplicates_the_same_csp_thesis() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                trade_status="Research",
                sell_leg_bid=1.1,
                sell_leg_ask=1.3,
                sell_leg_mid=1.2,
                quote_width_pct=0.08,
                short_oi=900,
                short_volume=120,
            ),
            _candidate(
                trade_status="Research",
                sell_leg_bid=1.1,
                sell_leg_ask=1.3,
                sell_leg_mid=1.2,
                quote_width_pct=0.10,
                short_oi=850,
                short_volume=110,
            ),
        ]
    )

    board = build_opportunity_board(
        scored=scored,
        final=pd.DataFrame(),
        watchlist=pd.DataFrame(),
        portfolio={"status": "ok", "cash": 25_000},
    )
    wheel = board[board["Lane"].eq("Wheel/Cash")]

    assert len(wheel) == 1
    assert wheel.iloc[0]["Trade"] == "Cash-secured put (assignment-risk): sell AAA 2026-05-29 100P"


def test_momentum_debit_lane_surfaces_actual_debit_spread_not_placeholder() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                ticker="BBB",
                direction="Bull Call",
                strategy="Bull Call Debit Spread",
                trade_status="Research",
                debit=2.0,
                mid_debit=2.0,
                natural_debit=2.15,
                max_profit=300.0,
                max_loss=200.0,
                short_leg="BBB260529C00105000",
                long_leg="BBB260529C00100000",
            )
        ]
    )

    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok", "cash": 25_000})
    debit = board[board["Lane"].eq("Momentum Debit")].iloc[0]

    assert debit["Ticker"] == "BBB"
    assert "Bull Call Debit Spread: buy BBB 2026-05-29 100C / sell BBB 2026-05-29 105C" == debit["Trade"]
    assert "No Momentum Debit setup qualified" not in debit["Trade"]


def test_fallback_income_discovers_wider_otm_weekly_credit_candidate() -> None:
    stock = pd.DataFrame(
        [
            {
                "ticker": "INTC",
                "close": 118.96,
                "sector": "Technology",
                "iv30d": 1.1256,
                "iv_rank": 85.8,
                "flow_bias": 0.07,
                "flow_total_premium": 505_000_000,
                "next_earnings_dt": dt.date(2026, 7, 23),
            }
        ]
    )
    hot = pd.DataFrame(
        [
            {"ticker": "INTC", "right": "P", "expiry_dt": dt.date(2026, 7, 17), "dte": 58, "strike": 90.0, "option_symbol": "INTC260717P00090000", "volume": 605, "open_interest": 3057, "bid": 3.75, "ask": 3.90},
            {"ticker": "INTC", "right": "P", "expiry_dt": dt.date(2026, 7, 17), "dte": 58, "strike": 95.0, "option_symbol": "INTC260717P00095000", "volume": 706, "open_interest": 3720, "bid": 5.00, "ask": 5.20},
            {"ticker": "INTC", "right": "P", "expiry_dt": dt.date(2026, 7, 17), "dte": 58, "strike": 100.0, "option_symbol": "INTC260717P00100000", "volume": 1040, "open_interest": 5501, "bid": 6.50, "ask": 6.75},
        ]
    )
    signals = {
        "top_flow_universe": pd.DataFrame(
            [
                {
                    "rank": 10,
                    "ticker": "INTC",
                    "flow_direction": "bullish",
                    "rank_score": 0.98,
                    "volume_oi_ratio": 1.05,
                    "total_premium": 505_000_000,
                    "vwap_confirmation": "bullish_above_tape_vwap",
                }
            ]
        )
    }

    out = build_fallback_income_candidates(stock_screener=stock, hot_chains=hot, liquidity_shift=signals, asof=ASOF)

    assert out.iloc[0]["ticker"] == "INTC"
    assert out.iloc[0]["direction"] == "Bull Put"
    assert out.iloc[0]["short_leg_eod"] == "INTC260717P00095000"
    assert out.iloc[0]["long_leg_eod"] == "INTC260717P00090000"
    assert out.iloc[0]["target_entry"] == 1.40


def test_fallback_income_work_limit_does_not_become_execute_below_target() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                construction_source="fallback_income",
                fallback_target_credit=1.40,
                target_entry=1.40,
                credit=1.10,
                mid_credit=1.33,
                natural_credit=1.10,
                quote_width_pct=0.04,
                short_oi=3720,
                short_volume=706,
                long_oi=3057,
                long_volume=605,
                live_status="PASS",
                hard_rejects="",
                penalties="too_close_to_expected_move;replay_guard_bull_put_expected_move",
            )
        ]
    )

    out = apply_fallback_income_status(scored)

    assert out.iloc[0]["trade_status"] == "Watch"
    assert out.iloc[0]["trade_tier"] == "fallback-income-work-limit"
    assert "do not mark Execute" in out.iloc[0]["trade_status_reason"]
    assert "replay_guard_bull_put_expected_move" not in out.iloc[0]["penalties"]


def test_fallback_income_can_execute_when_live_credit_meets_target() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                construction_source="fallback_income",
                fallback_target_credit=1.40,
                target_entry=1.40,
                credit=1.45,
                mid_credit=1.47,
                natural_credit=1.42,
                quote_width_pct=0.04,
                short_oi=3720,
                short_volume=706,
                long_oi=3057,
                long_volume=605,
                live_status="PASS",
                hard_rejects="",
                penalties="too_close_to_expected_move",
            )
        ]
    )

    out = apply_fallback_income_status(scored)

    assert out.iloc[0]["trade_status"] == "Execute"
    assert out.iloc[0]["trade_tier"] == "Execute Fallback Income"
    assert out.iloc[0]["required_entry"] == 1.40


def test_fallback_income_non_anchor_live_alternative_stays_work_limit() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                construction_source="fallback_income",
                live_construction_source="better_credit",
                fallback_target_credit=1.40,
                target_entry=1.40,
                credit=2.00,
                mid_credit=2.22,
                natural_credit=1.90,
                quote_width_pct=0.04,
                short_oi=3720,
                short_volume=706,
                long_oi=3057,
                long_volume=605,
                live_status="PASS",
                hard_rejects="",
                penalties="",
            )
        ]
    )

    out = apply_fallback_income_status(scored)

    assert out.iloc[0]["trade_status"] == "Watch"
    assert out.iloc[0]["trade_tier"] == "fallback-income-work-limit"
    assert "not the original flow-anchored fallback" in out.iloc[0]["trade_status_reason"]


def test_lifecycle_monitor_text_generated_for_execute_and_scout() -> None:
    board = pd.DataFrame(
        [
            {
                "Lane": "Execute",
                "Status": "🟢 Execute",
                "Ticker": "AAA",
                "Trade": "Bull Put Credit Spread",
                "Expiry": str(EXPIRY),
                "Entry limit": ">= $1.20 credit",
                "Max profit": "$120.00",
                "Max loss": "$380.00",
                "credit": 1.2,
                "short_strike": 100.0,
                "short_delta": -0.22,
                "dte": 10,
            },
            {
                "Lane": "Scout",
                "Status": "🔵 Scout",
                "Ticker": "BBB",
                "Trade": "Bull Call Debit Spread",
                "Expiry": str(EXPIRY),
                "Entry limit": "<= $2.00 debit",
                "Max profit": "$300.00",
                "Max loss": "$200.00",
                "debit": 2.0,
                "short_strike": 105.0,
                "short_delta": 0.25,
                "dte": 10,
            },
        ]
    )

    out = apply_lifecycle_triggers(board, asof=ASOF)

    assert out["phone_alert_text"].str.contains("Manual order only").all()
    assert out["profit_take"].astype(str).str.len().gt(0).all()
    assert out["short_leg_delta_threshold"].astype(str).str.contains("delta").all()


def test_target_feasibility_uses_expected_value_and_position_risk() -> None:
    board = pd.DataFrame(
        [
            {
                "Lane": "Execute",
                "Status": "🟢 Execute",
                "Ticker": "AAA",
                "direction": "Bull Put",
                "Target profit": "$500.00",
                "Max loss": "$1,000.00",
                "contracts": 2,
                "position_max_loss": 2_000.0,
                "target_profit_total": 1_000.0,
                "expected_value_total": 150.0,
            }
        ]
    )

    model = build_v3_target_model(
        asof=dt.date(2026, 5, 19),
        board=board,
        monthly_profit_target=10_000,
        month_to_date_realized_pnl=1_000,
        risk_budget=5_000,
        available_cash=10_000,
        live_outcome_status="ok",
        live_outcome_count=50,
        live_outcome_profit_factor=1.40,
    )

    assert model["remaining_monthly_target"] == 9000
    assert model["required_daily_pl"] > 0
    assert model["target_gap"]["risk_required"] is not None
    assert model["current_qualified_opportunity_expected_pl"] == 150.0
    assert model["current_qualified_opportunity_target_profit"] == 1000.0
    assert model["current_qualified_opportunity_max_loss"] == 2000.0
    assert model["target_feasibility"] == "infeasible"


def test_target_feasibility_is_not_demonstrated_without_closed_live_evidence() -> None:
    board = pd.DataFrame(
        [
            {
                "Lane": "Execute",
                "Status": "🟢 Execute",
                "Ticker": "AAA",
                "direction": "Bull Put",
                "Target profit": "$500.00",
                "Max loss": "$1,000.00",
                "position_max_loss": 1_000.0,
                "target_profit_total": 500.0,
                "expected_value_total": 100.0,
            }
        ]
    )

    model = build_v3_target_model(asof=ASOF, board=board, risk_budget=5_000)

    assert model["target_feasibility"] == "not demonstrated"
    assert model["risk_inputs"]["live_target_evidence_ok"] is False


def test_loss_review_downgrades_similar_recent_losing_setup() -> None:
    ledger = pd.DataFrame(
        [
            {
                "report_date": "2026-05-10",
                "ticker": "AAA",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "realized_pnl": -250.0,
            }
        ]
    )
    review = review_recent_losses(ledger, asof=ASOF)
    scored = pd.DataFrame([_candidate()])

    out = apply_loss_review(scored, review)

    assert "recent_loss_family:credit spreads" in out["penalties"].iloc[0]
    assert out["score"].iloc[0] < scored["score"].iloc[0]


def test_no_trade_audit_classifies_zero_execute_reason() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Research", live_status="chain_error", hard_rejects="chain_error")])
    board = build_opportunity_board(scored=scored, final=pd.DataFrame(), watchlist=pd.DataFrame(), portfolio={"status": "ok"})

    audit = classify_no_trade_audit(board=board, scored=scored, data_quality={"status": "ok", "critical_blockers": []}, portfolio={"status": "ok"})

    assert audit["classification"] == "data failure"
    assert "Schwab" in audit["exact_blocker"]


def test_missed_opportunity_audit_classifies_later_working_rejections() -> None:
    ledger = pd.DataFrame(
        [
            {
                "asof": "2026-05-10",
                "ticker": "AAA",
                "lane": "Research/Avoid",
                "status": "🟡 Research",
                "strategy": "Bull Put Credit Spread",
                "realized_pnl": 150.0,
                "mfe": 220.0,
                "thesis_worked": True,
                "reason_for_win_loss": "news_unconfirmed later cleared",
            },
            {
                "asof": "2026-05-10",
                "ticker": "BBB",
                "lane": "Execute",
                "status": "🟢 Execute",
                "strategy": "Bull Put Credit Spread",
                "realized_pnl": 100.0,
            },
        ]
    )

    audit = build_missed_opportunity_audit(ledger)

    assert audit["ticker"].tolist() == ["AAA"]
    assert audit["classification"].iloc[0] == "bad data/news gap"


def test_macro_event_gates_capture_named_calendar_risks(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-19"
    browser_dir = base_dir / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-macro.txt").write_text(
        "Fed minutes tomorrow. CPI risk faded. Jobs report due Friday.",
        encoding="utf-8",
    )

    gates = build_macro_event_gates(
        base_dir=base_dir,
        asof=ASOF,
        stock_screener=pd.DataFrame([{"ticker": "AAA", "next_earnings_dt": "2026-05-22"}]),
        regime={"trend": "downtrend", "volatility": "medium", "flow": "weak", "vix_proxy": 18.0},
    )

    observed = gates[gates["status"].eq("observed")]
    assert {"Fed", "CPI", "jobs", "major earnings", "market regime"}.issubset(set(observed["gate"]))


def test_all_lanes_write_to_v3_recommendation_ledger(tmp_path) -> None:
    board = pd.DataFrame(
        [
            {"Lane": "Execute", "Status": "🟢 Execute", "Ticker": "AAA", "Trade": "Bull Put Credit Spread", "Expiry": "2026-05-29", "recommended_limit": 1.2},
            {"Lane": "Scout", "Status": "🔵 Scout", "Ticker": "BBB", "Trade": "Bear Call Credit Spread", "Expiry": "2026-05-29", "recommended_limit": 1.1},
            {"Lane": "Momentum Debit", "Status": "🟡 Research", "Ticker": "CCC", "Trade": "Bull Call Debit Spread", "Expiry": "2026-05-29"},
            {"Lane": "Index/ETF", "Status": "🟡 Research", "Ticker": "SPY", "Trade": "Bear Call Credit Spread", "Expiry": "2026-05-29"},
            {"Lane": "Portfolio Repair", "Status": "🟡 Repair", "Ticker": "AAA", "Trade": "ROLL short option", "Expiry": ""},
            {"Lane": "Wheel/Cash", "Status": "🟡 Research", "Ticker": "CASH", "Trade": "CSP search", "Expiry": ""},
        ]
    )

    run_path, _ = write_recommendation_ledger(tmp_path / "codexdaily_v3_2026-05-19", ASOF, board)
    ledger = pd.read_csv(run_path)

    assert set(ledger["lane"]) >= {"Execute", "Scout", "Momentum Debit", "Index/ETF", "Portfolio Repair", "Wheel/Cash"}


def test_v3_recommendation_ledger_preserves_outcomes_and_distinct_structures(tmp_path) -> None:
    out_dir = tmp_path / "codexdaily_v3_2026-05-19"
    first = pd.DataFrame(
        [
            {
                "Lane": "Execute",
                "Status": "🟢 Execute",
                "Ticker": "AAA",
                "Trade": "Sell 95P / Buy 90P",
                "Expiry": "2026-05-29",
                "recommended_limit": 1.25,
            }
        ]
    )
    _, global_path = write_recommendation_ledger(out_dir, ASOF, first)
    ledger = pd.read_csv(global_path)
    ledger.loc[0, "actual_fill"] = 1.30
    ledger.loc[0, "close_fill"] = 0.50
    ledger.loc[0, "realized_pnl"] = 80.0
    ledger.loc[0, "outcome_status"] = "CLOSED"
    ledger.to_csv(global_path, index=False)

    rerun = pd.concat(
        [
            first,
            pd.DataFrame(
                [
                    {
                        "Lane": "Execute",
                        "Status": "🟢 Execute",
                        "Ticker": "AAA",
                        "Trade": "Sell 94P / Buy 89P",
                        "Expiry": "2026-05-29",
                        "recommended_limit": 1.25,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_recommendation_ledger(out_dir, ASOF, rerun)
    updated = pd.read_csv(global_path)

    assert updated["trade_key"].nunique() == 2
    closed = updated[updated["trade"].eq("Sell 95P / Buy 90P")].iloc[0]
    assert closed["realized_pnl"] == 80.0
    assert closed["outcome_status"] == "CLOSED"


def test_missed_opportunity_audit_classifies_later_worked_research() -> None:
    ledger = pd.DataFrame(
        [
            {
                "asof": "2026-05-19",
                "ticker": "AAA",
                "lane": "Research/Avoid",
                "status": "🟡 Research",
                "strategy": "Bull Put Credit Spread",
                "realized_pnl": 150.0,
                "mfe": 180.0,
                "thesis_worked": True,
                "reason_for_win_loss": "later positive outcome was not promoted",
            }
        ]
    )

    audit = build_missed_opportunity_audit(ledger)

    assert audit["classification"].tolist() == ["over-filtering"]


def test_snapshot_summary_file_is_written(tmp_path) -> None:
    chain_dir = tmp_path / "schwab_chains"
    chain_dir.mkdir()
    (chain_dir / "AAA.json").write_text(
        json.dumps(
            {
                "underlyingPrice": 100.0,
                "putExpDateMap": {
                    "2026-05-29:10": {
                        "100.0": [
                            {
                                "symbol": "AAA  260529P00100000",
                                "strikePrice": 100.0,
                                "bid": 1.1,
                                "ask": 1.3,
                                "mark": 1.2,
                                "delta": -0.22,
                                "volatility": 35.0,
                                "openInterest": 1000,
                                "totalVolume": 100,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    path = summarize_schwab_chain_snapshots(tmp_path, ASOF)
    summary = pd.read_csv(path)

    assert path.exists()
    assert summary["ticker"].tolist() == ["AAA"]
    assert summary["bid"].iloc[0] == 1.1


def test_overlay_mode_updates_prior_analysis_and_explains_changes(tmp_path) -> None:
    prior = tmp_path / "prior"
    out_dir = tmp_path / "overlay"
    prior.mkdir()
    pd.DataFrame([_candidate(trade_status="Research", oi_carryover_status="no_exact_match")]).to_csv(
        prior / "codexdaily_v3_scored_2026-05-19.csv",
        index=False,
    )
    overlay = tmp_path / "chain-oi-changes-2026-05-20.csv"
    pd.DataFrame(
        [
            {
                "option_symbol": "AAA260529P00100000",
                "oi_diff_plain": 250,
                "prev_bid_volume": 1000,
                "prev_ask_volume": 100,
                "prev_total_premium": 1_000_000,
            },
            {
                "option_symbol": "AAA260529P00095000",
                "oi_diff_plain": 100,
                "prev_bid_volume": 500,
                "prev_ask_volume": 50,
                "prev_total_premium": 500_000,
            },
            {
                "option_symbol": "",
                "oi_diff_plain": 1,
                "prev_bid_volume": 1,
                "prev_ask_volume": 1,
                "prev_total_premium": 1,
            },
        ]
    ).to_csv(overlay, index=False)

    manifest = run_overlay(prior_out_dir=prior, overlay_file=overlay, out_dir=out_dir, asof=ASOF, overlay_date=dt.date(2026, 5, 20))
    changes = pd.read_csv(out_dir / "codexdaily_v3_overlay_changes_2026-05-19_2026-05-20.csv")

    assert manifest["pipeline_name"] == "Codex Daily V3"
    assert not changes.empty
    assert "supportive" in set(changes["new_oi_support_or_conflict"])
    assert changes["exact_reason"].astype(str).str.len().gt(0).all()


def test_validation_harness_uses_systematic_recent_source_complete_dates(tmp_path) -> None:
    for name in ["2026-05-17", "2026-05-18", "2026-05-19"]:
        folder = tmp_path / name
        folder.mkdir()
        for prefix in ["stock-screener-", "hot-chains-", "bot-eod-report-"]:
            (folder / f"{prefix}{name}.csv").write_text("ticker\nAAA\n", encoding="utf-8")
    overlay_like = tmp_path / "2026-05-19-v3-overlay-2026-05-20-live"
    overlay_like.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "bot-eod-report-"]:
        (overlay_like / f"{prefix}2026-05-19.csv").write_text("ticker\nBBB\n", encoding="utf-8")
    incomplete = tmp_path / "2026-05-20"
    incomplete.mkdir()
    (incomplete / "stock-screener-2026-05-20.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    selected = select_systematic_date_folders(tmp_path, as_of=ASOF, latest_n=2)

    assert [path.name for path in selected] == ["2026-05-19", "2026-05-18"]


def test_validation_accepts_dp_only_bundle_as_explicit_degraded_history(tmp_path) -> None:
    folder = tmp_path / "2026-03-23"
    folder.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "dp-eod-report-"]:
        (folder / f"{prefix}2026-03-23.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    selected = select_systematic_date_folders(tmp_path, as_of=dt.date(2026, 3, 23), latest_n=5)

    assert selected == [folder]


def _write_bot_report(base_dir: Path, rows: list[dict]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(base_dir / f"bot-eod-report-{ASOF}.csv", index=False)


def _flow_row(ticker: str, minute: int, premium: float, **overrides) -> dict:
    row = {
        "executed_at": f"2026-05-19T14:{minute:02d}:00Z",
        "underlying_symbol": ticker,
        "side": "ask",
        "strike": 100,
        "option_type": "call",
        "expiry": str(EXPIRY),
        "underlying_price": 101.0,
        "size": 1,
        "premium": premium,
        "volume": 10,
        "open_interest": 100,
        "delta": 0.35,
        "gamma": 0.01,
        "report_flags": "",
        "upstream_condition_detail": "",
        "canceled": "f",
        "sector": "Technology",
    }
    row.update(overrides)
    return row


def test_flow_velocity_catches_child_order_accumulation(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-19"
    rows = [_flow_row("AAA", minute, 150_000) for minute in [0, 1, 2, 3, 4, 5]]
    _write_bot_report(base_dir, rows)

    signals = build_liquidity_shift_signals(
        base_dir=base_dir,
        root=tmp_path,
        asof=ASOF,
        stock_screener=pd.DataFrame([{"ticker": "AAA", "close": 101, "prev_close": 100, "total_open_interest": 10_000, "avg30_volume": 1_000_000}]),
        regime={"vix_proxy": 12.0, "volatility": "low"},
    )
    velocity = signals["flow_velocity"]

    assert bool(velocity.iloc[0]["child_order_accumulation"]) is True
    assert velocity.iloc[0]["rolling_15m_premium"] >= 900_000


def test_top_50_net_flow_sweep_expands_beyond_fixed_universe(tmp_path) -> None:
    base_dir = tmp_path / "2026-05-19"
    rows = [_flow_row(f"T{i:02d}", 0, 10_000 + i * 10_000) for i in range(60)]
    _write_bot_report(base_dir, rows)
    screener = pd.DataFrame(
        [
            {"ticker": f"T{i:02d}", "close": 50 + i, "prev_close": 49 + i, "total_open_interest": 5_000 + i, "avg30_volume": 500_000 + i}
            for i in range(60)
        ]
    )

    signals = build_liquidity_shift_signals(base_dir=base_dir, root=tmp_path, asof=ASOF, stock_screener=screener, regime={"vix_proxy": 18.0})
    top = signals["top_flow_universe"]

    assert len(top) == 50
    assert "T59" in set(top["ticker"])
    assert "uw_discovered" in set(top["source"])


def test_correlation_anomaly_flags_ticker_index_divergence(tmp_path) -> None:
    closes = [
        ("2026-05-15", 100, 500),
        ("2026-05-16", 101, 501),
        ("2026-05-17", 102, 502),
        ("2026-05-18", 103, 503),
        ("2026-05-19", 112, 504),
    ]
    for name, aaa_close, spy_close in closes:
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame(
            [
                {"ticker": "AAA", "close": aaa_close, "prev_close": aaa_close - 1, "bullish_premium": 1, "bearish_premium": 0, "next_earnings_date": ""},
                {"ticker": "SPY", "close": spy_close, "prev_close": spy_close - 1, "bullish_premium": 1, "bearish_premium": 0, "next_earnings_date": ""},
            ]
        ).to_csv(folder / f"stock-screener-{name}.csv", index=False)
    top_flow = pd.DataFrame([{"ticker": "AAA", "sector": "Unknown", "net_premium": 2_000_000, "flow_direction": "bullish"}])

    anomalies = build_correlation_anomalies(top_flow, tmp_path, asof=ASOF)

    assert not anomalies.empty
    assert bool(anomalies.iloc[0]["anomaly"]) is True
    assert anomalies.iloc[0]["benchmark"] == "SPY"


def test_volatility_regime_changes_unusual_flow_thresholds() -> None:
    low = compute_volatility_thresholds({"vix_proxy": 12.0})
    high = compute_volatility_thresholds({"vix_proxy": 31.0})

    assert low["premium_5m_threshold"] < high["premium_5m_threshold"]
    assert "lowers" in low["why"]
    assert "raises" in high["why"]


def test_zero_dte_gamma_engine_emits_pinning_context_when_data_exists() -> None:
    events = pd.DataFrame(
        [
            _flow_row("SPY", 0, 300_000, expiry=str(ASOF), strike=650, underlying_price=650.5, option_type="call", side="ask", volume=2000, open_interest=5000, gamma=0.02),
            _flow_row("SPY", 1, 290_000, expiry=str(ASOF), strike=650, underlying_price=650.4, option_type="put", side="ask", volume=1900, open_interest=5200, gamma=0.02),
        ]
    )
    events["executed_at"] = pd.to_datetime(events["executed_at"], utc=True)
    events["expiry_dt"] = pd.to_datetime(events["expiry"]).dt.date
    events["ticker"] = events["underlying_symbol"]
    events["abs_premium"] = events["premium"].abs()
    events["signed_premium"] = [300_000, -290_000]

    gamma = build_zero_dte_gamma_context(events, asof=ASOF)

    assert gamma["ticker"].tolist() == ["SPY"]
    assert gamma.iloc[0]["setup_type"] in {"pinning", "directional", "reversal", "liquidity-trap"}
    assert gamma.iloc[0]["pinning_level"] == 650


def test_intraday_tier2_requires_vwap_confirmation() -> None:
    scored = pd.DataFrame([_candidate(ticker="AAA", direction="Bull Put")])
    signals = {
        "top_flow_universe": pd.DataFrame(
            [
                {
                    "rank": 20,
                    "ticker": "AAA",
                    "rank_score": 0.8,
                    "flow_direction": "bullish",
                    "vwap_confirmation": "bullish_vwap_not_confirmed",
                }
            ]
        ),
        "flow_velocity": pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "rolling_5m_premium": 800_000,
                    "rolling_15m_premium": 1_700_000,
                    "child_order_accumulation": False,
                    "flow_velocity_signal": True,
                }
            ]
        ),
    }

    out = apply_liquidity_shift_context(scored, signals, require_intraday_vwap=True)

    assert out.iloc[0]["alpha_tier"] == "Tier 2"
    assert "vwap_unconfirmed_tier2_intraday" in out.iloc[0]["hard_rejects"]


def test_tier2_cannot_size_like_tier1() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                alpha_tier="Tier 2",
                max_loss=100.0,
                max_profit=90.0,
                target_profit_total=90.0,
                expected_value_per_contract=40.0,
                contracts=1,
            )
        ]
    )

    final = select_final_trades(
        scored,
        regime={"sizing_stance": "normal"},
        risk_budget=10_000,
        recent_performance={"status": "ok"},
        max_final_trades=1,
        risk_config={"risk_mandate": "target-growth", "max_contracts_per_trade": 20, "minimum_expected_value_per_dollar_risk": 0.01},
    )

    assert int(final.iloc[0]["contracts"]) == 1
    assert "Tier 2" in final.iloc[0]["sizing_rationale"]
