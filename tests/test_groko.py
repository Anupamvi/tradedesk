from __future__ import annotations

import datetime as dt

import pandas as pd

import groko
import groko.core as groko_core
from groko._vendor import schwab_live as groko_schwab
from uwos import options_agent as oa
from uwos.options_agent import core as oa_core


def test_groko_is_forked_from_options_agent_not_codex() -> None:
    assert groko.PIPELINE_NAME == "Groko"
    assert groko.PIPELINE_VERSION == "groko-v1.6-send-now-recovery-20260827"
    assert groko.DEFAULT_OUTPUT_NAMESPACE == "groko"
    assert groko_core.PREVIOUS_PIPELINE_VERSIONS[0] == "groko-v1.5-eod-quote-session-20260827"
    assert groko_core.PREVIOUS_PIPELINE_VERSIONS[1] == "groko-v1.4-live-send-now-parity-20260827"
    assert groko_core.assert_strict_goal_runtime_defaults() is None
    assert groko_core.__file__ != oa_core.__file__
    assert oa.PIPELINE_NAME == "Options Agent"
    assert oa.PIPELINE_VERSION == "options-agent-v1.84-construction-cap-parity-20260826-210000"


def test_groko_has_no_codex_daily_v2_import() -> None:
    import groko.core as core
    import groko.replay as replay

    assert "codexuw" not in getattr(core, "__dict__", {})
    assert not any(name == "codexuw" or name.startswith("codexuw.") for name in core.__dict__)
    source = core.__file__
    text = open(source, encoding="utf-8").read()
    assert "import codexuw" not in text
    assert "from codexuw" not in text
    replay_text = open(replay.__file__, encoding="utf-8").read()
    assert "import codexuw" not in replay_text
    assert "from codexuw" not in replay_text


def test_groko_keeps_v184_selector_execution_contract() -> None:
    policy = groko_core._promoted_selector_policy_definition()
    assert groko_core._promoted_selector_daily_cap() == 35
    assert groko_core.DEFAULT_LIVE_CHAIN_TICKER_CAP == 35
    assert policy["source_quote_width_policy"] == "live_parity"
    assert policy["credit_expected_move_gate"] is False
    assert groko_core.MIN_SEND_NOW_CREDIT_WIDTH_RATIO == 0.25
    assert groko_core.MIN_SEND_NOW_CREDIT == 0.50
    source = open(groko_core.__file__, encoding="utf-8").read()
    assert "min_actionable_credit_width_ratio=MIN_SEND_NOW_CREDIT_WIDTH_RATIO" in source
    assert "setdefault(\"min_actionable_credit_width_ratio\", MIN_SEND_NOW_CREDIT_WIDTH_RATIO)" in source or "setdefault('min_actionable_credit_width_ratio', MIN_SEND_NOW_CREDIT_WIDTH_RATIO)" in source
    source = open(groko_schwab.__file__, encoding="utf-8").read()
    assert '_expected_move_ratio"] >= 0.75' not in source
    assert '["credit_pct_width", "liq_score"' in source or '"_risk_sized"' in source


def test_groko_writes_own_artifacts_not_codex_names() -> None:
    from pathlib import Path

    paths = groko_core.output_paths("2026-08-26", root=Path("/tmp"), out_dir=Path("/tmp/groko-test"))
    assert paths["manifest"].name == "groko_manifest_2026-08-26.json"
    assert paths["report"].name == "groko_report_2026-08-26.md"
    assert paths["groko_trades"].name == "GROKO_TRADES.md"
    assert paths["portfolio_context"].name == "groko_portfolio_context.json"
    out = groko_core.default_output_dir(Path("/tmp/desk"), "2026-08-26")
    assert out.as_posix().endswith("/out/groko/2026-08-26")


def _send_now_credit_row(**overrides):
    row = {
        "ticker": "NEM",
        "issue_type": "Common Stock",
        "strategy_route": "bear_call_credit",
        "entry_type": "CREDIT",
        "underlying_quality_tier": "core",
        "regime": "mixed",
        "dte": 23,
        "credit_width_ratio": 0.25,
        "entry_credit": 1.25,
        "target_entry": 1.25,
        "combined_flow_bias": -0.08,
        "live_distance_pct": 0.08,
        "live_expected_move_pct": 0.08,
        "live_quote_width_pct": 0.10,
        "source_contract_volume": 141.0,
        "source_contract_oi": 2_498.0,
        "synthesis_score": 80.0,
        "hard_rejects": "",
        "earnings_within_holding_horizon": False,
    }
    row.update(overrides)
    return row


def test_credit_quality_does_not_hard_reject_rank_only_flow() -> None:
    rejects = groko_core._trade_quality_rejects(
        entry_credit=1.25,
        credit_width_ratio=0.25,
        max_loss=375.0,
        signal_premium=3_000_000,
        combined_flow_bias=-0.08,
    )
    assert rejects == []
    assert not any(item.startswith("directional_bias_below_") for item in rejects)

    junk = groko_core._trade_quality_rejects(
        entry_credit=0.05,
        credit_width_ratio=0.05,
        max_loss=2_000,
        signal_premium=500_000,
        combined_flow_bias=0.01,
    )
    assert "entry_credit_below_0.25" in junk
    assert "credit_width_ratio_below_16pct" in junk
    assert "one_lot_max_loss_above_750" in junk
    assert not any(item.startswith("directional_bias_below_") for item in junk)


def test_missing_quote_is_wait_not_pretend_wide() -> None:
    policy = groko_core._promoted_selector_policy_definition()
    missing, _, reasons, _ = groko_core._selector_policy_row_assessment(
        _send_now_credit_row(
            live_quote_width_pct=None,
            dated_combo_quote_width_pct=None,
            entry_quote_width_pct=None,
        ),
        policy=policy,
    )
    assert missing is False
    assert "credit_quote_width_unavailable" in reasons
    assert "credit_quote_width_above_0.25" not in reasons

    wide, _, reasons, _ = groko_core._selector_policy_row_assessment(
        _send_now_credit_row(live_quote_width_pct=0.40),
        policy=policy,
    )
    assert wide is False
    assert "credit_quote_width_above_0.25" in reasons


def test_directional_bias_hard_reject_does_not_veto_credit_selector() -> None:
    policy = groko_core._promoted_selector_policy_definition()
    eligible, _, reasons, _ = groko_core._selector_policy_row_assessment(
        _send_now_credit_row(hard_rejects="directional_bias_below_0.10"),
        policy=policy,
    )
    assert eligible is True
    assert "objective_quality_reject" not in reasons


def test_live_chain_cap_reserves_send_now_width_credits(monkeypatch) -> None:
    monkeypatch.delenv("UWOS_OPTIONS_AGENT_LIVE_CHAIN_TICKER_CAP", raising=False)
    priced = pd.DataFrame(
        [
            _send_now_credit_row(
                ticker="NEM",
                hard_rejects="directional_bias_below_0.10",
                live_quote_width_pct=None,
                source_contract_volume=141.0,
            ),
            _send_now_credit_row(
                ticker="RBLX",
                source_contract_volume=2.0,
                source_contract_oi=29.0,
                live_quote_width_pct=None,
            ),
            *[
                {
                    "ticker": f"CHEAP{idx:02d}",
                    "issue_type": "Common Stock",
                    "strategy_route": "bear_call_credit",
                    "entry_type": "CREDIT",
                    "underlying_quality_tier": "core",
                    "dte": 20,
                    "credit_width_ratio": 0.16,
                    "entry_credit": 0.50,
                    "source_contract_volume": 100.0,
                    "source_contract_oi": 1_000.0,
                    "hard_rejects": "",
                    "earnings_within_holding_horizon": False,
                }
                for idx in range(groko_core.DEFAULT_LIVE_CHAIN_TICKER_CAP)
            ],
        ]
    )

    allowed, cap, deferred = groko_core._allowed_live_validation_tickers(
        priced,
        enabled=True,
    )

    assert "NEM" in allowed
    assert "RBLX" in allowed
    assert cap >= 2
    assert groko_core._send_now_credit_needs_live_chain(priced.iloc[0].to_dict())
    assert groko_core._send_now_credit_needs_live_chain(priced.iloc[1].to_dict())


def test_bull_put_stays_unproven_when_heldout_pf_fails_independent_bar() -> None:
    def build_rows(route: str, pre_count: int, heldout_pnls: list[float]) -> list[dict]:
        rows: list[dict] = []
        for offset in range(pre_count):
            rows.append(
                {
                    "strategy_route": route,
                    "selector_partition": "pre_split",
                    "signal_date": pd.Timestamp("2026-01-02") + pd.Timedelta(days=offset),
                    "ticker": f"{route[:3].upper()}{offset}",
                    "pnl_1x": 50.0,
                    "realized_pnl": 50.0,
                    "return_on_risk": 0.10,
                    "next_session_reprice_observed": True,
                    "next_session_reprice_approved": True,
                    "exact_evaluated": True,
                }
            )
        for offset, pnl in enumerate(heldout_pnls):
            rows.append(
                {
                    "strategy_route": route,
                    "selector_partition": "heldout_test",
                    "signal_date": pd.Timestamp("2026-05-04") + pd.Timedelta(days=offset),
                    "ticker": f"{route[:3].upper()}H{offset}",
                    "pnl_1x": pnl,
                    "realized_pnl": pnl,
                    "return_on_risk": pnl / 500.0,
                    "next_session_reprice_observed": True,
                    "next_session_reprice_approved": True,
                    "exact_evaluated": True,
                }
            )
        return rows

    frame = pd.DataFrame(
        build_rows("bear_call_credit", 30, [50.0] * 15)
        + build_rows("bull_put_credit", 30, [10.0] * 8 + [-20.0] * 7)
    )
    supported = groko_core._active_selector_supported_routes(frame)
    assert supported == {"bear_call_credit"}


def test_debit_stays_off_execute_until_both_partitions_pass() -> None:
    assert groko_core.BULL_CALL_DEBIT_SELECTOR_POLICY_ID in groko_core.DEBIT_SELECTOR_POLICY_IDS
    assert groko_core.BEAR_PUT_DEBIT_SELECTOR_POLICY_ID in groko_core.DEBIT_SELECTOR_POLICY_IDS
    promoted = groko_core._promoted_selector_policy_definition()
    assert "DEBIT" not in {
        groko_core._as_text(value).upper()
        for value in promoted.get("allowed_entry_types", ())
    }


def test_missing_next_session_leg_quote_does_not_veto_profitable_debit() -> None:
    policy = groko_core._selector_policy_definition(
        groko_core.BULL_CALL_DEBIT_SELECTOR_POLICY_ID
    )
    rows = []
    for offset in range(30):
        rows.append(
            {
                "strategy_route": "bull_call_debit",
                "signal_date": pd.Timestamp("2026-01-02") + pd.Timedelta(days=offset),
                "ticker": f"BC{offset}",
                "realized_pnl": 40.0,
                "return_on_risk": 0.15,
                "next_session_reprice_observed": True,
                "next_session_reprice_approved": True,
                "next_session_reprice_reason": "",
                "exact_evaluated": True,
            }
        )
    for offset in range(20):
        missing = offset >= 12
        pnl = 50.0 if not missing else float("nan")
        rows.append(
            {
                "strategy_route": "bull_call_debit",
                "signal_date": pd.Timestamp("2026-05-04") + pd.Timedelta(days=offset),
                "ticker": f"BCH{offset}",
                "realized_pnl": pnl,
                "return_on_risk": 0.20 if not missing else float("nan"),
                "next_session_reprice_observed": not missing,
                "next_session_reprice_approved": not missing,
                "next_session_reprice_reason": (
                    "missing_next_session_entry_leg_quote" if missing else ""
                ),
                "exact_evaluated": not missing,
            }
        )
    heldout = pd.DataFrame(rows[-20:])
    metrics = groko_core._selector_partition_metrics(
        heldout,
        policy=policy,
        partition="heldout_test",
        source_path="test",
    )
    assert metrics["sample_size"] == 12
    assert metrics["reprice_observed_coverage"] == 1.0
    assert metrics["outcome_coverage"] == 1.0
    assert metrics["partition_status"] == "PASS"


def test_after_hours_mid_is_a_live_quote_when_natural_is_crossed() -> None:
    alts = groko_core._natural_price_spread_alternatives(
        [
            {
                "live_status": "PASS",
                "spread_width": 5.0,
                "natural_credit": -0.05,
                "mid_credit": 1.25,
                "credit": 1.25,
                "short_strike": 100.0,
                "expected_move_pct": 0.08,
            }
        ],
        entry_type="CREDIT",
        direction="Bull Put",
        spot=110.0,
    )
    assert alts[0]["live_status"] == "PASS"
    assert alts[0]["pricing_policy"] == "after_hours_mid"
    assert alts[0]["credit"] == 1.25
    assert alts[0]["credit_pct_width"] == 0.25

    dead = groko_core._natural_price_spread_alternatives(
        [
            {
                "live_status": "PASS",
                "spread_width": 5.0,
                "natural_credit": -0.05,
                "mid_credit": 0.0,
                "credit": 0.0,
                "short_strike": 100.0,
            }
        ],
        entry_type="CREDIT",
        direction="Bull Put",
        spot=110.0,
    )
    assert dead[0]["live_status"] == "NO_EXECUTABLE_NATURAL_PRICE"


def test_after_hours_schwab_quote_is_not_discarded_for_transient_width() -> None:
    preserved = groko_core._preserve_target_when_closed_market_quote_is_transiently_bad(
        {"trade_plan": "SELL 1 AAA 100 Put / BUY 1 AAA 95 Put @ 1.25 CREDIT", "entry_limit": 1.25, "max_profit": 125, "max_loss": 375},
        {"quality_gate_reason": "live_quote_width_pct_above_30pct", "hard_rejects": ""},
    )
    assert preserved is None


def test_groko_color_board_uses_status_colors() -> None:
    empty = groko_core.build_color_trade_board_markdown(pd.DataFrame(), as_of="2026-08-27")
    assert empty.startswith("# Groko trades — 2026-08-27")
    assert "🟢 send-now" in empty
    assert "🔴 **No tickets.**" in empty

    tickets = pd.DataFrame(
        [
            {
                "ticker": "PLTR",
                "structure": "Bull Put Credit Spread",
                "expiry": "2026-09-18",
                "dte": 23,
                "entry_limit": 1.40,
                "suggested_contracts": 5,
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 PLTR 150 Put / BUY 1 PLTR 145 Put @ 1.40 CREDIT",
                "position_max_profit": 700,
                "position_max_loss": 1800,
                "account_risk_pct": 0.002,
                "execution_blockers": "",
                "selector_policy_reason": "",
            },
            {
                "ticker": "NOW",
                "structure": "Bear Call Credit Spread",
                "expiry": "2026-09-18",
                "dte": 23,
                "entry_limit": 1.25,
                "suggested_contracts": 1,
                "ready_to_enter": False,
                "target_order_status": "not_actionable",
                "trade_plan": "SELL 1 NOW 900 Call / BUY 1 NOW 905 Call @ 1.25 CREDIT",
                "position_max_profit": 125,
                "position_max_loss": 375,
                "account_risk_pct": 0.001,
                "execution_blockers": "selector_policy_block",
                "selector_policy_reason": "credit_width_outside_0.25_0.30",
            },
        ]
    )
    board = groko_core.build_color_trade_board_markdown(tickets, as_of="2026-08-27")
    assert "🟢 actionable: **1**" in board
    pltr = board.index("| 🟢 | **PLTR**")
    now = board.index("| 🔴 | **NOW**")
    assert pltr < now


def test_debit_verticals_reserve_after_hours_live_chains() -> None:
    row = {
        "ticker": "TSLA",
        "issue_type": "Common Stock",
        "strategy_route": "bull_call_debit",
        "entry_type": "DEBIT",
        "dte": 21,
        "entry_limit": 1.80,
        "target_entry": 1.80,
    }
    assert groko_core._debit_vertical_needs_live_chain(row) is True


def test_selector_block_is_review_not_avoid() -> None:
    status = groko_core._selector_constrained_recommendation_status(
        groko_core.RecommendationStatus.ENTER.value,
        "BLOCK",
    )
    assert status == groko_core.RecommendationStatus.REVIEW.value
    still_avoid = groko_core._selector_constrained_recommendation_status(
        groko_core.RecommendationStatus.AVOID.value,
        "BLOCK",
    )
    assert still_avoid == groko_core.RecommendationStatus.AVOID.value


def test_objective_quality_reject_is_not_an_opaque_bucket() -> None:
    echo = {
        "hard_rejects": "directional_bias_below_0.10; external_agent_objective_blocker; one_lot_max_loss_above_750"
    }
    assert groko_core._selector_blocking_hard_rejects(echo) == []
    veto = {"hard_rejects": "credit_width_ratio_below_16pct; external_agent_objective_blocker"}
    assert groko_core._selector_blocking_hard_rejects(veto) == ["credit_width_ratio_below_16pct"]

    policy = groko_core._promoted_selector_policy_definition()
    eligible, _, reasons, _ = groko_core._selector_policy_row_assessment(
        _send_now_credit_row(hard_rejects="directional_bias_below_0.10; external_agent_objective_blocker"),
        policy=policy,
    )
    assert eligible is True
    assert "objective_quality_reject" not in reasons


def test_quality_notes_do_not_avoid_a_live_structure() -> None:
    status, hard, reason, quality, _note = groko_core._finalize_quality_rejects(
        groko_core.RecommendationStatus.ENTER.value,
        [
            "directional_bias_below_0.10",
            "live_quote_width_pct_above_30pct",
            "external_agent_objective_blocker",
        ],
        note="live validated",
    )
    assert status == groko_core.RecommendationStatus.ENTER.value
    assert hard == ""
    assert quality == "reviewable"
    assert "live_quote_width_pct_above_30pct" in reason

    status, hard, reason, quality, _note = groko_core._finalize_quality_rejects(
        groko_core.RecommendationStatus.ENTER.value,
        ["credit_width_ratio_below_16pct", "directional_bias_below_0.10"],
        note="dated",
    )
    assert status == groko_core.RecommendationStatus.AVOID.value
    assert hard == "credit_width_ratio_below_16pct"


def test_live_validated_selector_block_reaches_review_tickets() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "PLTR",
                "strategy_route": "bull_put_credit",
                "entry_type": "CREDIT",
                "structure": "Bull Put Credit Spread",
                "recommendation_status": groko_core.RecommendationStatus.ENTER.value,
                "selector_policy_status": "BLOCK",
                "selector_policy_reason": "active_selector_route_not_proven_in_frozen_replay",
                "quality_status": "qualified",
                "trade_quality_status": "reviewable",
                "hard_rejects": "",
                "quality_gate_reason": "",
                "live_validation_status": "PASS",
                "entry_limit": 1.40,
                "target_entry": 1.40,
                "suggested_contracts": 1,
                "max_profit": 140,
                "max_loss": 360,
                "credit_width_ratio": 0.28,
                "underlying_quality_tier": "core",
                "trade_plan": "SELL 1 PLTR 2026-09-18 170 Put / BUY 1 PLTR 2026-09-18 165 Put @ 1.40 CREDIT",
                "dte": 23,
                "expiry": "2026-09-18",
            }
        ]
    )
    decision = groko_core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    row = decision.iloc[0]
    assert row["final_action"] == groko_core.RecommendationStatus.REVIEW.value
    assert row["execution_status"] != "blocked"
    assert row["target_order_status"] == "review_only_selector_not_selected"
    tickets = groko_core.build_trade_tickets(decision)
    assert not tickets.empty
    assert tickets.iloc[0]["ticker"] == "PLTR"


def test_mixed_credit_sleeve_keeps_nonlosing_bull_put_route() -> None:
    def rows(route: str, partition: str, start: str, count: int, pnl: float) -> list[dict]:
        return [
            {
                "strategy_route": route,
                "selector_partition": partition,
                "signal_date": pd.Timestamp(start) + pd.Timedelta(days=offset),
                "ticker": f"{route[:3].upper()}{offset}",
                "realized_pnl": pnl,
                "return_on_risk": 0.05,
                "next_session_reprice_observed": True,
                "next_session_reprice_approved": True,
                "exact_evaluated": True,
            }
            for offset in range(count)
        ]

    frame = pd.DataFrame(
        rows("bear_call_credit", "pre_split", "2026-01-02", 30, 50.0)
        + rows("bear_call_credit", "heldout_test", "2026-05-04", 15, 20.0)
        + rows("bull_put_credit", "pre_split", "2026-01-02", 30, 40.0)
        + rows("bull_put_credit", "heldout_test", "2026-05-04", 15, 2.0)
    )
    supported = groko_core._active_selector_supported_routes(frame)
    assert supported == {"bear_call_credit", "bull_put_credit"}


def test_live_pass_uses_live_pair_volume_not_dated_zero() -> None:
    volume = groko_core._selector_contract_volume(
        {
            "live_validation_status": "PASS",
            "dated_source_contract_volume": 0.0,
            "source_contract_volume": 0.0,
            "live_short_volume": 80.0,
            "live_long_volume": 50.0,
        }
    )
    assert volume == 50.0
    overnight = groko_core._selector_contract_volume(
        {
            "live_validation_status": "PASS",
            "dated_source_contract_volume": 91.0,
            "source_contract_volume": 0.0,
            "live_short_volume": 0.0,
            "live_long_volume": 0.0,
        }
    )
    assert overnight == 91.0
    partial_session = groko_core._selector_contract_volume(
        {
            "live_validation_status": "PASS",
            "dated_source_contract_volume": 91.0,
            "source_contract_volume": 0.0,
            "live_short_volume": 8.0,
            "live_long_volume": 8.0,
        }
    )
    assert partial_session == 91.0


def test_live_quote_accepts_prior_regular_session() -> None:
    latest = groko_core._latest_observable_regular_market_session_date()
    prior = groko_core._latest_regular_market_day_on_or_before(latest - dt.timedelta(days=1))
    allowed = groko_core._acceptable_live_quote_sessions(prior)
    assert prior in allowed
    assert latest in allowed
    oi = groko_core._selector_contract_open_interest(
        {
            "live_validation_status": "PASS",
            "dated_source_contract_oi": 10.0,
            "source_contract_oi": 10.0,
            "live_short_oi": 400.0,
            "live_long_oi": 220.0,
        }
    )
    assert oi == 220.0


def test_eod_uw_date_accepts_next_session_schwab_quotes() -> None:
    """Yesterday's UW folder + today's live stamps must not STALE the send-now sleeve.

    The 2026-08-26 live book selected NOW 29% / PLTR / MSTR / INTC, then marked
    108/165 chains STALE because --date 2026-08-26 and quote session 2026-08-27
    were an exact mismatch. That zeros greens every morning.
    """

    asof = dt.date(2026, 8, 26)
    after_open = dt.datetime(2026, 8, 27, 6, 33, tzinfo=groko_core.MARKET_TIME_ZONE)
    before_open = dt.datetime(2026, 8, 27, 2, 0, tzinfo=groko_core.MARKET_TIME_ZONE)
    assert groko_core._live_quote_sessions_are_fresh({dt.date(2026, 8, 27)}, asof, now=after_open)
    assert groko_core._live_quote_sessions_are_fresh({dt.date(2026, 8, 26)}, asof, now=after_open)
    assert groko_core._live_quote_sessions_are_fresh({dt.date(2026, 8, 26)}, asof, now=before_open)
    assert groko_core._live_quote_sessions_are_fresh({dt.date(2026, 8, 27)}, asof, now=before_open)
    # One newer far-dated stamp must not veto an otherwise current chain.
    assert groko_core._live_quote_sessions_are_fresh(
        {dt.date(2026, 8, 26), dt.date(2026, 8, 27), dt.date(2026, 8, 28)},
        asof,
        now=after_open,
    )
    assert not groko_core._live_quote_sessions_are_fresh({dt.date(2026, 8, 21)}, asof, now=after_open)


def test_selector_pass_generic_actual_route_does_not_block_green() -> None:
    """Promoted-selector proof outranks a mixed-book actual_route cohort.

    NOW 08-26 was selector PASS / calibration PASS (model held-out PF 1.75) and
    only blocked by STALE. Generic bear_call actuals (n=9, PF 0.35) must stay
    diagnostic or the sleeve can never print a green.
    """

    row = {
        "selector_policy_status": "PASS",
        "profitability_calibration_status": "PASS",
        "profitability_calibration_scope": "actual_route",
        "profitability_calibration_actual_status": "WARN",
        "profitability_calibration_actual_sample_size": 9,
        "profitability_calibration_actual_avg_pnl": -18.67,
        "profitability_calibration_actual_profit_factor": 0.354,
        "profitability_calibration_replay_status": "WARN",
        "profitability_calibration_replay_sample_size": 19,
        "profitability_calibration_replay_avg_pnl": -18.86,
        "profitability_calibration_replay_profit_factor": 0.709,
        "profitability_calibration_route_replay_status": "PASS",
        "profitability_calibration_route_replay_sample_size": 767,
        "profitability_calibration_route_replay_avg_pnl": 17.95,
        "profitability_calibration_route_replay_profit_factor": 2.262,
        "profitability_calibration_model_replay_status": "PASS",
        "profitability_calibration_model_replay_sample_size": 191,
        "profitability_calibration_model_replay_avg_pnl": 45.13,
        "profitability_calibration_model_replay_profit_factor": 2.858,
        "profitability_calibration_model_exact_replay_status": "WARN",
        "profitability_calibration_model_exact_replay_sample_size": 17,
        "profitability_calibration_model_exact_replay_avg_pnl": 12.46,
        "profitability_calibration_model_exact_replay_profit_factor": 1.225,
        "profitability_calibration_model_replay_pre_split_sample_size": 118,
        "profitability_calibration_model_replay_pre_split_avg_pnl": 55.78,
        "profitability_calibration_model_replay_pre_split_profit_factor": 4.4,
        "profitability_calibration_model_replay_heldout_sample_size": 73,
        "profitability_calibration_model_replay_heldout_avg_pnl": 27.91,
        "profitability_calibration_model_replay_heldout_profit_factor": 1.754,
        "actual_forward_strategy_expectancy_status": "WARN",
        "actual_forward_strategy_expectancy_sample_size": 9,
        "actual_forward_strategy_expectancy_avg_pnl": -18.67,
        "actual_forward_strategy_expectancy_profit_factor": 0.354,
        "actual_forward_strategy_expectancy_scope": "strategy_route",
        "actual_forward_expectancy_status": "BLOCK",
        "actual_forward_expectancy_sample_size": 0,
    }
    assert groko_core._active_selector_model_bridge_makes_generic_actual_diagnostic(row)
    assert not groko_core._negative_strategy_expectancy_blocks_green(row)
    assert groko_core._partitioned_selector_model_route_ticket_ready(row, require_selector=True)
    assert groko_core._positive_strategy_expectancy_ready_for_green(row)


def test_sleeve_selected_bull_put_heldout_one_oh_prints_one_lot_green() -> None:
    """Mixed-sleeve bull puts are the live 25-30% names. Independent held-out
    PF 1.01 is not a losing book; requiring 1.20 here repeats 0-green days.
    Generic actual_route from mixed wheel/OA books stays diagnostic.
    """

    row = {
        "selector_policy_status": "PASS",
        "profitability_calibration_status": "PASS",
        "profitability_calibration_scope": "actual_route",
        "profitability_calibration_actual_status": "WARN",
        "profitability_calibration_actual_sample_size": 20,
        "profitability_calibration_actual_avg_pnl": -73.25,
        "profitability_calibration_actual_profit_factor": 0.195,
        "profitability_calibration_replay_status": "WARN",
        "profitability_calibration_replay_sample_size": 7,
        "profitability_calibration_replay_avg_pnl": 110.54,
        "profitability_calibration_replay_profit_factor": float("inf"),
        "profitability_calibration_route_replay_status": "PASS",
        "profitability_calibration_route_replay_sample_size": 481,
        "profitability_calibration_route_replay_avg_pnl": 23.43,
        "profitability_calibration_route_replay_profit_factor": 2.485,
        "profitability_calibration_model_replay_status": "PASS",
        "profitability_calibration_model_replay_sample_size": 134,
        "profitability_calibration_model_replay_avg_pnl": 45.0,
        "profitability_calibration_model_replay_profit_factor": 2.716,
        "profitability_calibration_model_exact_replay_status": "WARN",
        "profitability_calibration_model_exact_replay_sample_size": 18,
        "profitability_calibration_model_exact_replay_avg_pnl": 59.51,
        "profitability_calibration_model_exact_replay_profit_factor": 3.513,
        "profitability_calibration_model_replay_pre_split_sample_size": 88,
        "profitability_calibration_model_replay_pre_split_avg_pnl": 68.14,
        "profitability_calibration_model_replay_pre_split_profit_factor": 8.501,
        "profitability_calibration_model_replay_heldout_sample_size": 46,
        "profitability_calibration_model_replay_heldout_avg_pnl": 0.73,
        "profitability_calibration_model_replay_heldout_profit_factor": 1.012,
        "actual_forward_strategy_expectancy_status": "WARN",
        "actual_forward_strategy_expectancy_sample_size": 20,
        "actual_forward_strategy_expectancy_avg_pnl": -73.25,
        "actual_forward_strategy_expectancy_profit_factor": 0.195,
        "actual_forward_strategy_expectancy_scope": "strategy_route",
        "actual_forward_expectancy_status": "WARN",
        "actual_forward_expectancy_sample_size": 1,
        "actual_forward_expectancy_avg_pnl": -10.0,
        "actual_forward_expectancy_profit_factor": 0.5,
    }
    assert groko_core._model_only_selector_route_calibration_ready(
        actual_scope="actual_route",
        actual_metrics={
            "status": "WARN",
            "sample_size": 20,
            "avg_pnl": -73.25,
            "profit_factor": 0.195,
        },
        replay_metrics={
            "status": "WARN",
            "sample_size": 7,
            "avg_pnl": 110.54,
            "profit_factor": float("inf"),
        },
        route_replay_metrics={
            "status": "PASS",
            "sample_size": 481,
            "avg_pnl": 23.43,
            "profit_factor": 2.485,
        },
        model_replay_metrics={
            "status": "PASS",
            "sample_size": 134,
            "avg_pnl": 45.0,
            "profit_factor": 2.716,
        },
        model_pre_split_metrics={
            "sample_size": 88,
            "avg_pnl": 68.14,
            "profit_factor": 8.501,
        },
        model_heldout_metrics={
            "sample_size": 46,
            "avg_pnl": 0.73,
            "profit_factor": 1.012,
        },
        selector_exact_replay_metrics={
            "status": "WARN",
            "sample_size": 18,
            "avg_pnl": 59.51,
            "profit_factor": 3.513,
        },
    )
    assert groko_core._active_selector_model_bridge_makes_generic_actual_diagnostic(row)
    assert not groko_core._negative_strategy_expectancy_blocks_green(row)
    assert groko_core._partitioned_selector_model_route_ticket_ready(row, require_selector=True)
    assert groko_core._positive_strategy_expectancy_ready_for_green(row)


def test_selector_exact_loss_still_blocks_one_lot_green() -> None:
    ready = groko_core._model_only_selector_route_calibration_ready(
        actual_scope="actual_route",
        actual_metrics={"status": "WARN", "sample_size": 20, "avg_pnl": -73.25, "profit_factor": 0.195},
        replay_metrics={"status": "WARN", "sample_size": 2, "avg_pnl": -289.1, "profit_factor": 0.0},
        route_replay_metrics={"status": "PASS", "sample_size": 481, "avg_pnl": 23.43, "profit_factor": 2.485},
        model_replay_metrics={"status": "PASS", "sample_size": 134, "avg_pnl": 45.0, "profit_factor": 2.716},
        model_pre_split_metrics={"sample_size": 88, "avg_pnl": 68.14, "profit_factor": 8.501},
        model_heldout_metrics={"sample_size": 46, "avg_pnl": 0.73, "profit_factor": 1.012},
        selector_exact_replay_metrics={"status": "WARN", "sample_size": 5, "avg_pnl": -2.4, "profit_factor": 0.967},
    )
    assert ready is False


def test_credit_send_now_economics_rejects_compressed_width() -> None:
    assert groko_core._credit_send_now_economics_ok(
        {
            "entry_type": "CREDIT",
            "strategy_route": "bear_call_credit",
            "entry_limit": 1.30,
            "credit_width_ratio": 0.26,
            "live_quote_width_pct": 0.07,
        }
    )
    assert not groko_core._credit_send_now_economics_ok(
        {
            "entry_type": "CREDIT",
            "strategy_route": "bear_call_credit",
            "entry_limit": 0.97,
            "credit_width_ratio": 0.194,
            "live_quote_width_pct": 0.12,
        }
    )
    assert not groko_core._credit_send_now_economics_ok(
        {
            "entry_type": "CREDIT",
            "strategy_route": "bear_call_credit",
            "entry_limit": 1.35,
            "credit_width_ratio": 0.27,
            "live_quote_width_pct": 0.36,
        }
    )


def test_bull_call_prefers_macro_safe_dte_window() -> None:
    minimum_dte, maximum_dte = groko_core._debit_vertical_preferred_dte_window("Bull Call")
    assert minimum_dte == groko_core.MIN_MACRO_SCALE_DTE
    assert maximum_dte == 45
    bear_put = groko_core._debit_vertical_preferred_dte_window("Bear Put")
    assert bear_put == (14, 45)


def test_sleeve_send_now_scales_to_initial_order_cap() -> None:
    row = {
        "selector_policy_status": "PASS",
        "live_validation_status": "PASS",
        "entry_type": "CREDIT",
        "strategy_route": "bull_put_credit",
        "entry_limit": 1.30,
        "credit_width_ratio": 0.26,
        "live_quote_width_pct": 0.07,
        "dte": 22,
        "profitability_calibration_status": "PASS",
        "profitability_calibration_scope": "actual_route",
        "profitability_calibration_actual_status": "WARN",
        "profitability_calibration_actual_sample_size": 20,
        "profitability_calibration_actual_avg_pnl": -73.25,
        "profitability_calibration_actual_profit_factor": 0.195,
        "profitability_calibration_replay_status": "WARN",
        "profitability_calibration_replay_sample_size": 7,
        "profitability_calibration_replay_avg_pnl": 110.54,
        "profitability_calibration_replay_profit_factor": float("inf"),
        "profitability_calibration_route_replay_status": "PASS",
        "profitability_calibration_route_replay_sample_size": 481,
        "profitability_calibration_route_replay_avg_pnl": 23.43,
        "profitability_calibration_route_replay_profit_factor": 2.485,
        "profitability_calibration_model_replay_status": "PASS",
        "profitability_calibration_model_replay_sample_size": 134,
        "profitability_calibration_model_replay_avg_pnl": 45.0,
        "profitability_calibration_model_replay_profit_factor": 2.716,
        "profitability_calibration_model_exact_replay_status": "WARN",
        "profitability_calibration_model_exact_replay_sample_size": 18,
        "profitability_calibration_model_exact_replay_avg_pnl": 59.51,
        "profitability_calibration_model_exact_replay_profit_factor": 3.513,
        "profitability_calibration_model_replay_pre_split_sample_size": 88,
        "profitability_calibration_model_replay_pre_split_avg_pnl": 68.14,
        "profitability_calibration_model_replay_pre_split_profit_factor": 8.501,
        "profitability_calibration_model_replay_heldout_sample_size": 46,
        "profitability_calibration_model_replay_heldout_avg_pnl": 0.73,
        "profitability_calibration_model_replay_heldout_profit_factor": 1.012,
    }
    assert groko_core._sleeve_send_now_scale_allowed(row)
    cap = groko_core._sleeve_send_now_size_cap(row, risk_sized_contracts=9)
    assert cap == groko_core.PROMOTED_SELECTOR_INITIAL_ORDER_CONTRACTS
    cheapened = dict(row)
    cheapened["credit_width_ratio"] = 0.194
    cheapened["entry_limit"] = 0.97
    assert groko_core._sleeve_send_now_size_cap(cheapened, risk_sized_contracts=9) is None
