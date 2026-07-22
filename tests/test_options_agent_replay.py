import datetime as dt
import hashlib
import json
import math

import pandas as pd
import pytest

from uwos.exact_spread_backtester import LegQuote
from uwos.options_agent import core
from uwos.options_agent import replay


def _quote(bid: float, ask: float) -> LegQuote:
    return LegQuote(
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        volume=500.0,
        open_interest=2_000.0,
        source_kind="test",
    )


def test_replay_uses_conservative_entry_and_exit_sides() -> None:
    short = _quote(2.00, 2.20)
    long = _quote(0.80, 1.00)

    credit_bid, credit_ask, credit_mid = replay._spread_quotes("CREDIT", short, long)
    debit_bid, debit_ask, debit_mid = replay._spread_quotes("DEBIT", short, long)

    assert (credit_bid, credit_ask, credit_mid) == pytest.approx((1.0, 1.4, 1.2))
    assert replay._entry_price("CREDIT", credit_bid, credit_ask) == pytest.approx(1.0)
    assert replay._exit_value("CREDIT", credit_bid, credit_ask) == pytest.approx(1.4)
    assert (debit_bid, debit_ask, debit_mid) == pytest.approx((-1.4, -1.0, -1.2))
    assert replay._entry_price("DEBIT", debit_bid, debit_ask) == pytest.approx(-1.0)
    assert replay._exit_value("DEBIT", debit_bid, debit_ask) == pytest.approx(-1.4)


def test_replay_pnl_includes_round_trip_commission() -> None:
    assert replay._pnl("CREDIT", 1.20, 0.60) == 57.40
    assert replay._pnl("DEBIT", 1.20, 1.80) == 57.40


def test_replay_exit_market_applies_vertical_no_arbitrage_bounds() -> None:
    assert replay._bounded_exit_market(-0.20, 5.40, 5.0) == (0.0, 5.0)
    assert replay._bounded_exit_market(1.20, 1.00, 5.0) is None


def test_replay_management_levels_and_triggers_match_order_policy() -> None:
    assert replay._management_levels("CREDIT", 1.0, 5.0) == (0.35, 2.0)
    assert replay._management_trigger("CREDIT", 0.30, 0.35, 2.0, final_session=False) == "take_profit"
    assert replay._management_trigger("CREDIT", 2.10, 0.35, 2.0, final_session=False) == "stop_loss"
    assert replay._management_levels("DEBIT", 1.0, 5.0) == (1.8, 0.5)
    assert replay._management_trigger("DEBIT", 0.40, 1.8, 0.5, final_session=False) == "stop_loss"
    assert replay._management_trigger("DEBIT", 1.00, 1.8, 0.5, final_session=True) == "time_exit"


def test_replay_horizon_stays_bound_to_live_holding_horizon() -> None:
    assert replay.FIXED_HORIZON_SESSIONS == core.PLANNED_TRADE_HOLDING_SESSIONS


def test_replay_uses_next_session_reprice_without_changing_source_selection_fields() -> None:
    signal_day = dt.date(2026, 7, 21)
    entry_day = dt.date(2026, 7, 22)
    exit_day = dt.date(2026, 7, 28)
    short = core._human_option_leg_symbol_key(
        ticker="AAA",
        expiry="2026-08-21",
        strike=100.0,
        option_type="PUT",
    )
    long = core._human_option_leg_symbol_key(
        ticker="AAA",
        expiry="2026-08-21",
        strike=95.0,
        option_type="PUT",
    )
    assert short and long
    row = {
        "ticker": "AAA",
        "trade_plan": "SELL 1 AAA 2026-08-21 100 Put / BUY 1 AAA 2026-08-21 95 Put @ 1.00 CREDIT",
        "structure": "bull put credit spread",
        "strategy_route": "bull_put_credit",
        "entry_type": "CREDIT",
        "expiry": "2026-08-21",
        "dte": 31,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "close": 110.0,
        "iv30d": 0.30,
        "quality_status": "qualified",
        "trade_quality_status": "reviewable",
        "recommendation_status": "ENTER",
        "score": 80.0,
        "target_entry": 1.00,
        "signal_premium": 1_000_000.0,
        "combined_flow_bias": 0.50,
    }
    target_quotes = {short: _quote(2.00, 2.20), long: _quote(0.80, 1.00)}
    worse_next_session_quotes = {short: _quote(1.90, 2.10), long: _quote(1.20, 1.40)}

    unfilled = replay._replay_row(
        row,
        signal_day=signal_day,
        entry_day=entry_day,
        exit_day=exit_day,
        regime="risk_on",
        target_quote_index=target_quotes,
        next_session_quote_index=worse_next_session_quotes,
    )

    assert unfilled["exact_fillable"] is True
    assert unfilled["target_entry_limit"] == pytest.approx(1.00)
    assert unfilled["entry_price"] == pytest.approx(1.00)
    assert unfilled["source_contract_volume"] == pytest.approx(500.0)
    assert unfilled["next_session_reprice_observed"] is True
    assert unfilled["next_session_reprice_approved"] is False
    assert unfilled["next_session_reprice_reason"] == "next_session_credit_below_source_target"
    assert len(replay._selector_frame(pd.DataFrame([unfilled], columns=replay.DETAIL_COLUMNS))) == 1

    filled = replay._replay_row(
        row,
        signal_day=signal_day,
        entry_day=entry_day,
        exit_day=exit_day,
        regime="risk_on",
        target_quote_index=target_quotes,
        next_session_quote_index={short: _quote(2.20, 2.25), long: _quote(0.80, 0.85)},
    )

    assert filled["next_session_reprice_observed"] is True
    assert filled["next_session_reprice_approved"] is True
    assert filled["next_session_reprice_reason"] == "next_session_reprice_approved"
    assert filled["executed_entry_price"] == pytest.approx(1.35)
    assert filled["executed_entry_credit"] == pytest.approx(1.35)


def test_replay_marks_observed_invalid_net_market_as_resolved_no_entry() -> None:
    observed, approved, entry, reason = replay._next_session_reprice_status(
        {
            "expiry": "2026-08-21",
            "strategy_route": "bull_put_credit",
            "signal_premium": 1_000_000.0,
            "combined_flow_bias": 0.30,
        },
        entry_day=dt.date(2026, 7, 22),
        regime="risk_on",
        entry_type="CREDIT",
        target_limit=0.80,
        bid=-0.20,
        ask=-0.10,
        width=5.0,
        quote_width_pct=0.20,
        short_quote=_quote(0.50, 0.60),
        long_quote=_quote(0.70, 0.80),
    )

    assert observed is True
    assert approved is False
    assert entry is None
    assert reason == "invalid_next_session_reprice_economics"


def test_replay_reprice_enforces_the_live_selector_quote_width_cap() -> None:
    observed, approved, entry, reason = replay._next_session_reprice_status(
        {
            "expiry": "2026-08-21",
            "strategy_route": "bull_put_credit",
            "signal_premium": 1_000_000.0,
            "combined_flow_bias": 0.30,
        },
        entry_day=dt.date(2026, 7, 22),
        regime="risk_on",
        entry_type="CREDIT",
        target_limit=0.80,
        bid=1.00,
        ask=1.40,
        width=5.0,
        quote_width_pct=core.MAX_SELECTOR_ENTRY_QUOTE_WIDTH_PCT + 0.001,
        short_quote=_quote(2.00, 2.20),
        long_quote=_quote(0.80, 1.00),
    )

    assert observed is True
    assert approved is False
    assert entry == pytest.approx(1.00)
    assert reason == "next_session_reprice_quality_fail:selector_quote_width_above_0.25"


def test_replay_exit_observations_start_after_next_session_entry() -> None:
    assert replay._exit_observation_dates(
        dt.date(2026, 7, 22),
        dt.date(2026, 7, 28),
    ) == [
        dt.date(2026, 7, 23),
        dt.date(2026, 7, 24),
        dt.date(2026, 7, 27),
        dt.date(2026, 7, 28),
    ]


def test_replay_reprice_enforces_entry_day_dte_boundary() -> None:
    signal_day = dt.date(2026, 7, 21)
    short = core._human_option_leg_symbol_key(
        ticker="AAA",
        expiry="2026-07-28",
        strike=100.0,
        option_type="PUT",
    )
    long = core._human_option_leg_symbol_key(
        ticker="AAA",
        expiry="2026-07-28",
        strike=95.0,
        option_type="PUT",
    )
    assert short and long
    row = {
        "ticker": "AAA",
        "trade_plan": "SELL 1 AAA 2026-07-28 100 Put / BUY 1 AAA 2026-07-28 95 Put @ 1.00 CREDIT",
        "strategy_route": "bull_put_credit",
        "entry_type": "CREDIT",
        "expiry": "2026-07-28",
        "dte": 7,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "close": 110.0,
        "iv30d": 0.30,
        "quality_status": "qualified",
        "trade_quality_status": "reviewable",
        "recommendation_status": "ENTER",
        "target_entry": 1.0,
        "signal_premium": 1_000_000.0,
        "combined_flow_bias": 0.50,
    }
    quotes = {short: _quote(2.20, 2.25), long: _quote(0.80, 0.85)}

    replayed = replay._replay_row(
        row,
        signal_day=signal_day,
        entry_day=dt.date(2026, 7, 22),
        exit_day=dt.date(2026, 7, 28),
        regime="risk_on",
        target_quote_index=quotes,
        next_session_quote_index=quotes,
    )

    assert replayed["entry_dte"] == 6
    assert replayed["next_session_reprice_observed"] is True
    assert replayed["next_session_reprice_approved"] is False
    assert replayed["next_session_reprice_reason"] == "next_session_credit_dte_outside_7_45"


def test_replay_selector_uses_holding_horizon_for_earnings() -> None:
    policy = core.SELECTOR_CHALLENGER_POLICIES[0]
    row = {
        "asof": "2026-07-21",
        "strategy_route": "bear_call_credit",
        "entry_side": "CREDIT",
        "underlying_quality_tier": "core",
        "regime": "risk_on",
        "expiry": "2026-08-21",
        "next_earnings_dt": "2026-08-20",
        "dte": 30,
        "entry_quote_width_pct": 0.10,
        "combined_flow_bias": -0.30,
        "entry_credit_pct_width": 0.20,
        "expected_move_ratio": 1.0,
        "source_contract_volume": 100.0,
        "decision_score": 80.0,
        "macro_event_count_within_holding_horizon": 0,
    }

    eligible, _, reasons, _ = core._selector_v4_replay_assessment(row, policy=policy)

    assert eligible
    assert "earnings_within_holding_horizon" not in reasons

    eligible, _, reasons, _ = core._selector_v4_replay_assessment(
        {**row, "next_earnings_dt": "2026-07-24"},
        policy=policy,
    )

    assert not eligible
    assert "earnings_within_holding_horizon" in reasons


def test_replay_expected_move_ratio_uses_exact_fill_breakeven() -> None:
    row = {
        "close": 100.0,
        "dte": 35,
        "iv30d": 0.30,
        "strategy_route": "bull_put_credit",
        "short_strike": 92.0,
        "breakeven": 99.0,
    }

    ratio = replay._expected_move_ratio(row, "CREDIT", 1.0)

    expected_move = 0.30 * math.sqrt(35 / 365)
    assert ratio == pytest.approx(0.09 / expected_move)


def test_replay_decision_pass_uses_dated_quality_fields_only() -> None:
    baseline = {
        "quality_status": "qualified",
        "trade_quality_status": "reviewable",
        "recommendation_status": "REVIEW",
        "hard_rejects": "",
    }

    assert replay._dated_decision_pass(baseline)
    assert not replay._dated_decision_pass({**baseline, "recommendation_status": "AVOID"})
    assert not replay._dated_decision_pass({**baseline, "trade_quality_status": "rejected"})
    assert not replay._dated_decision_pass({**baseline, "hard_rejects": "negative credit"})


def test_replay_selector_preserves_underlying_quality_tier() -> None:
    row = {
        "strategy_route": "bull_put_credit",
        "entry_side": "CREDIT",
        "underlying_quality_tier": "speculative",
        "dte": 30,
        "entry_quote_width_pct": 0.10,
        "combined_flow_bias": 0.30,
        "entry_credit_pct_width": 0.20,
        "expected_move_ratio": 1.0,
        "source_contract_volume": 50.0,
        "decision_score": 80.0,
    }

    policy = core.SELECTOR_CHALLENGER_POLICIES[0]
    eligible, _, reasons, _ = core._selector_v4_replay_assessment(row, policy=policy)

    assert not eligible
    assert "underlying_not_core" in reasons

    quality_reject = {**row, "underlying_quality_tier": "core", "hard_rejects": "bad economics"}
    eligible, _, reasons, _ = core._selector_v4_replay_assessment(
        quality_reject,
        policy=policy,
    )
    assert not eligible
    assert "objective_quality_reject" in reasons


def test_next_session_replay_reuses_only_explicit_selector_only_candidate_cache() -> None:
    signal_day = dt.date(2026, 1, 2)
    row = {column: "" for column in replay.DETAIL_COLUMNS}
    row.update(
        {
            "asof": signal_day.isoformat(),
            "selected_for_policy": False,
            "exact_evaluated": False,
            "producer": "uwos.options_agent.replay",
        }
    )
    cached = pd.DataFrame([row], columns=replay.DETAIL_COLUMNS)
    audit = {
        "day": signal_day.isoformat(),
        "error": "",
        "required_source_status": "pass",
        "required_source_paths": {
            label: [f"/{label}-{signal_day.isoformat()}.zip"]
            for label in replay.REQUIRED_REPLAY_SOURCES
        },
        "cache_fingerprint": "legacy-entry-cache",
    }

    assert not replay._compatible_entry_cache(
        cached,
        audit,
        signal_day=signal_day,
        discovery_limit=None,
    )
    fingerprint = "75d950a8f8cc0e6daaa90de14271dbb369f6a4b04bb4eac98f4861cd42ed12b9"
    assert replay._compatible_entry_cache(
        cached,
        {**audit, "cache_fingerprint": fingerprint},
        signal_day=signal_day,
        discovery_limit=None,
    )
    cached.loc[0, "next_session_reprice_observed"] = False
    cached.loc[0, "next_session_reprice_reason"] = "invalid_next_session_reprice_economics"
    migrated, migration = replay._migrate_compatible_candidate_cache(
        cached,
        source_fingerprint=fingerprint,
    )
    assert migration == "v1_47_reprice_resolution_and_selector_quote_width_25pct"
    assert bool(migrated.loc[0, "next_session_reprice_observed"]) is True
    cached.loc[0, "next_session_reprice_observed"] = True
    cached.loc[0, "next_session_reprice_approved"] = True
    cached.loc[0, "next_session_quote_width_pct"] = 0.251
    migrated, _ = replay._migrate_compatible_candidate_cache(
        cached,
        source_fingerprint=fingerprint,
    )
    assert bool(migrated.loc[0, "next_session_reprice_approved"]) is False
    assert migrated.loc[0, "next_session_reprice_reason"] == (
        "next_session_reprice_quality_fail:selector_quote_width_above_0.25"
    )


def test_dated_construction_chain_unions_hot_and_chain_oi(monkeypatch, tmp_path) -> None:
    from codexuw import data as uw_data

    point_in_time_calls = []
    hot = pd.DataFrame(
        [
            {
                "option_symbol": "AAA260220C00100000",
                "ticker": "AAA",
                "right": "C",
                "expiry_dt": "2026-02-20",
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.2,
                "open_interest": 100,
                "volume": 10,
                "dte": 30,
            }
        ]
    )
    chain_oi = pd.DataFrame(
        [
            {
                "option_symbol": "AAA260220C00100000",
                "ticker": "AAA",
                "right": "C",
                "expiry_dt": "2026-02-20",
                "strike": 100.0,
                "last_bid": 0.9,
                "last_ask": 1.3,
                "curr_oi": 200,
                "volume": 20,
                "dte": 30,
            },
            {
                "option_symbol": "AAA260220C00105000",
                "ticker": "AAA",
                "right": "C",
                "expiry_dt": "2026-02-20",
                "strike": 105.0,
                "last_bid": 0.4,
                "last_ask": 0.5,
                "curr_oi": 150,
                "volume": 15,
                "dte": 30,
            },
        ]
    )
    def load_hot(*_args, **kwargs):
        point_in_time_calls.append(("hot", kwargs.get("point_in_time")))
        return hot

    def load_oi(*_args, **kwargs):
        point_in_time_calls.append(("oi", kwargs.get("point_in_time")))
        return chain_oi

    monkeypatch.setattr(uw_data, "load_hot_chains", load_hot)
    monkeypatch.setattr(uw_data, "load_chain_oi", load_oi)

    chain = core._dated_option_chain_for_construction(tmp_path, dt.date(2026, 1, 21))

    assert chain["option_symbol"].tolist() == [
        "AAA260220C00100000",
        "AAA260220C00105000",
    ]
    assert chain["dated_quote_source"].tolist() == ["hot_chain", "chain_oi"]
    assert chain.loc[chain["option_symbol"].eq("AAA260220C00105000"), "bid"].iloc[0] == 0.4
    assert point_in_time_calls == [("hot", True), ("oi", True)]


def test_raw_universe_reads_every_discovery_source_point_in_time(monkeypatch, tmp_path) -> None:
    from codexuw import data as uw_data

    day_dir = tmp_path / "2026-05-01"
    day_dir.mkdir()
    calls = []
    screener = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "close": 100.0,
                "prev_close": 99.0,
                "bullish_premium": 2_000_000.0,
                "bearish_premium": 500_000.0,
                "flow_total_premium": 2_500_000.0,
                "flow_bias": 0.6,
                "marketcap": 50_000_000_000.0,
                "issue_type": "Common Stock",
                "total_volume": 1_000_000.0,
                "avg30_volume": 900_000.0,
                "total_open_interest": 100_000.0,
            }
        ]
    )

    def load_screener(*_args, **kwargs):
        calls.append(("screener", kwargs.get("point_in_time")))
        return screener

    def load_hot(*_args, **kwargs):
        calls.append(("hot", kwargs.get("point_in_time")))
        return pd.DataFrame()

    def load_oi(*_args, **kwargs):
        calls.append(("oi", kwargs.get("point_in_time")))
        return pd.DataFrame()

    def load_bot(*_args, **kwargs):
        calls.append(("bot", kwargs.get("point_in_time")))
        return pd.DataFrame(columns=["ticker"])

    monkeypatch.setattr(uw_data, "load_stock_screener", load_screener)
    monkeypatch.setattr(uw_data, "load_hot_chains", load_hot)
    monkeypatch.setattr(uw_data, "load_chain_oi", load_oi)
    monkeypatch.setattr(uw_data, "aggregate_bot_flow", load_bot)

    universe, notes = core.build_raw_universe(day_dir, dt.date(2026, 5, 1))

    assert notes == []
    assert universe["ticker"].tolist() == ["AAA"]
    assert calls == [
        ("screener", True),
        ("hot", True),
        ("oi", True),
        ("bot", True),
    ]


def test_replay_day_fails_when_a_required_source_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        core,
        "build_source_inventory",
        lambda *_args, **_kwargs: {
            "sources": {
                "stock_screener": {"status": "present"},
                "hot_chains": {"status": "present"},
                "chain_oi": {"status": "missing"},
                "bot_eod": {"status": "present"},
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="chain_oi"):
        replay._candidate_rows_for_day(
            tmp_path,
            dt.date(2026, 5, 1),
            quote_store=None,
            discovery_limit=None,
        )


def test_selector_partition_blocks_incomplete_selected_outcomes() -> None:
    selected = pd.DataFrame(
        {
            "signal_date": pd.bdate_range("2026-01-02", periods=20),
            "ticker": [f"T{index:02d}" for index in range(20)],
            "strategy_route": ["bull_call_debit"] * 20,
            "realized_pnl": [100.0] * 18 + [math.nan, math.nan],
            "exact_evaluated": [True] * 18 + [False, False],
            "next_session_reprice_observed": [True] * 20,
            "next_session_reprice_approved": [True] * 20,
        }
    )

    row = core._selector_partition_metrics(
        selected,
        policy={"policy_id": "coverage_test"},
        partition="heldout_test",
        source_path="synthetic.csv",
    )

    assert row["selected_count"] == 20
    assert row["sample_size"] == 18
    assert row["outcome_coverage"] == 0.9
    assert row["partition_status"] == "BLOCK"
    assert "selected_execution_resolution_coverage_below_95pct" in row["blocking_reasons"]


def test_independent_replay_manifest_is_not_claimed_as_production_proof(tmp_path) -> None:
    detail = pd.DataFrame(columns=replay.DETAIL_COLUMNS)
    detail_path = tmp_path / "options_agent_replay_detail.csv"
    detail.to_csv(detail_path, index=False)

    metrics = replay._selected_metrics(detail)

    assert metrics["selected"] == 0
    assert metrics["outcome_coverage"] == 0.0
    assert replay.SCHEMA_VERSION == "options_agent.independent_replay.v4"


def test_replay_pin_rejects_capped_or_stale_manifest(tmp_path, monkeypatch) -> None:
    replay_dir = tmp_path / "out" / "options_agent_replay"
    replay_dir.mkdir(parents=True)
    detail_path = replay_dir / "options_agent_replay_detail.csv"
    manifest_path = replay_dir / "options_agent_replay_manifest.json"
    day_audit_path = replay_dir / "options_agent_replay_day_audit.csv"
    detail_path.write_text("ticker,pnl_1x\nAAA,10\n", encoding="utf-8")
    day_audit_path.write_text("day,error\n2026-01-02,\n", encoding="utf-8")
    valid = {
        "schema_version": replay.SCHEMA_VERSION,
        "producer": "uwos.options_agent.replay",
        "pipeline_version": core.PIPELINE_VERSION,
        "point_in_time_export_ceiling": True,
        "selection_outcome_independent": True,
        "production_discovery_parity": True,
        "required_source_labels": list(replay.REQUIRED_REPLAY_SOURCES),
        "optional_source_labels": list(replay.OPTIONAL_REPLAY_SOURCES),
        "optional_source_coverage": {"bot_eod": {"present_days": 5, "missing_days": 15}},
        "candidate_limit": 0,
        "max_days": 0,
        "days": 20,
        "successful_days": 20,
        "failed_days": 0,
        "source_coverage_status": "pass",
        "cache_fingerprint": "abc123",
        "day_audit_sha256": hashlib.sha256(day_audit_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(replay, "_cache_fingerprint", lambda _candidate_limit: "abc123")

    manifest_path.write_text(json.dumps(valid), encoding="utf-8")
    pin = replay.write_replay_pin(
        tmp_path,
        {"detail": detail_path, "manifest": manifest_path, "day_audit": day_audit_path},
        split_day=dt.date(2026, 5, 19),
    )
    assert json.loads(pin.read_text())["production_discovery_parity"] is True

    manifest_path.write_text(
        json.dumps({**valid, "candidate_limit": 25, "production_discovery_parity": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="capped, stale, or non-point-in-time"):
        replay.write_replay_pin(
            tmp_path,
            {"detail": detail_path, "manifest": manifest_path, "day_audit": day_audit_path},
            split_day=dt.date(2026, 5, 19),
        )


def test_replay_calendar_excludes_weekends_and_market_holidays() -> None:
    days = replay._eligible_replay_days(
        [
            dt.date(2026, 1, 16),
            dt.date(2026, 1, 17),
            dt.date(2026, 1, 18),
            dt.date(2026, 1, 19),
            dt.date(2026, 1, 20),
        ],
        start=dt.date(2026, 1, 16),
        end=dt.date(2026, 1, 27),
    )

    assert days == [dt.date(2026, 1, 16), dt.date(2026, 1, 20)]
