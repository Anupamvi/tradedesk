import datetime as dt
import json

import pytest
import pandas as pd

from uwos.options_agent import forward_registry
from uwos.options_agent import core
from uwos.options_agent.forward_registry import (
    BrokerMatchReason,
    ForwardRecommendationRegistry,
)


NOW = dt.datetime(2026, 7, 10, 18, 30, 0, 123456, tzinfo=dt.timezone.utc)
BUY_LEG = "SPY260717C00630000"
SELL_LEG = "SPY260717C00635000"


@pytest.fixture(autouse=True)
def _fixed_registration_clock(monkeypatch):
    monkeypatch.setattr(forward_registry, "_utc_now", lambda: NOW)


def _registry(tmp_path):
    return ForwardRecommendationRegistry(tmp_path / "forward_registry.jsonl")


def _legs(*, reversed=False):
    buy_side, sell_side = ("SELL", "BUY") if reversed else ("BUY", "SELL")
    return [
        {"side": buy_side, "ratio": 1, "occ_symbol": BUY_LEG},
        {"side": sell_side, "ratio": 1, "occ_symbol": SELL_LEG},
    ]


def _register(
    registry,
    *,
    logical_id="spy-call-spread",
    account_id="acct-123",
    status="GREEN",
    **overrides,
):
    values = {
        "logical_recommendation_id": logical_id,
        "account_id": account_id,
        "recommendation_date": NOW.date(),
        "status": status,
        "legs": _legs(),
        "code_provenance": {"git_commit": "abc123", "pipeline_version": "options-agent-test"},
        "run_provenance": {"run_id": "run-20260710-1", "source_date": "2026-07-10"},
        "live_current_date": True,
    }
    values.update(overrides)
    return registry.register(**values)


def test_identical_registration_is_idempotent_and_preserves_actual_utc_timestamp(tmp_path):
    registry = _registry(tmp_path)

    first = _register(registry)
    second = _register(registry)

    assert first == second
    assert first.registered_at == NOW
    assert first.registered_at.utcoffset() == dt.timedelta(0)
    assert first.code_provenance["git_commit"] == "abc123"
    assert first.run_provenance["run_id"] == "run-20260710-1"
    assert len(registry.events()) == 1
    assert len((tmp_path / "forward_registry.jsonl").read_text().splitlines()) == 1


def test_newer_downgrade_supersedes_green_without_rewriting_history(tmp_path):
    registry = _registry(tmp_path)
    green = _register(registry)

    downgrade = _register(registry, status="DOWNGRADED")

    assert green.is_active
    assert not downgrade.is_active
    assert [event.status for event in registry.events()] == ["GREEN", "DOWNGRADED"]
    assert registry.current_state() == (downgrade,)
    assert registry.current_active_state() == ()
    rows = [
        json.loads(line)
        for line in (tmp_path / "forward_registry.jsonl").read_text().splitlines()
    ]
    assert rows[0]["status"] == "GREEN"
    assert rows[1]["status"] == "DOWNGRADED"
    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW + dt.timedelta(seconds=1),
        legs=_legs(),
    )
    assert match.reason == BrokerMatchReason.NO_ACTIVE_RECOMMENDATION


def test_pre_registration_fill_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    _register(registry)

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW - dt.timedelta(microseconds=1),
        legs=_legs(),
    )

    assert not match.matched
    assert match.reason == BrokerMatchReason.PRE_REGISTRATION_FILL


def test_fill_after_quote_validity_window_is_not_matched(tmp_path):
    registry = _registry(tmp_path)
    _register(registry)

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW + dt.timedelta(minutes=16),
        legs=_legs(),
    )

    assert not match.matched
    assert match.reason == BrokerMatchReason.NO_ACTIVE_RECOMMENDATION


def test_later_downgrade_does_not_erase_an_already_valid_fill(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    green = _register(registry)
    fill_time = NOW + dt.timedelta(minutes=1)
    monkeypatch.setattr(
        forward_registry,
        "_utc_now",
        lambda: NOW + dt.timedelta(minutes=2),
    )
    _register(registry, status="DOWNGRADED")

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=fill_time,
        legs=_legs(),
    )

    assert match.matched
    assert match.recommendation == green


def test_reverse_directed_legs_are_rejected(tmp_path):
    registry = _registry(tmp_path)
    _register(registry)

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW + dt.timedelta(seconds=1),
        legs=_legs(reversed=True),
    )

    assert not match.matched
    assert match.reason == BrokerMatchReason.REVERSE_OR_DIFFERENT_LEGS


def test_account_mismatch_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    _register(registry, account_id="acct-123")

    match = registry.match_broker_fill(
        account_id="acct-999",
        fill_timestamp=NOW + dt.timedelta(seconds=1),
        legs=_legs(),
    )

    assert not match.matched
    assert match.reason == BrokerMatchReason.ACCOUNT_MISMATCH


def test_two_active_logical_recommendations_make_match_ambiguous(tmp_path):
    registry = _registry(tmp_path)
    _register(registry, logical_id="spy-call-spread-a")
    _register(registry, logical_id="spy-call-spread-b")

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW + dt.timedelta(seconds=1),
        legs=_legs(),
    )

    assert not match.matched
    assert match.reason == BrokerMatchReason.AMBIGUOUS_ACTIVE_RECOMMENDATIONS
    assert match.candidate_logical_ids == ("spy-call-spread-a", "spy-call-spread-b")


def test_backdated_registration_is_recorded_but_never_active(tmp_path):
    registry = _registry(tmp_path)

    event = _register(
        registry,
        recommendation_date=NOW.date() - dt.timedelta(days=1),
        live_current_date=True,
    )

    assert not event.eligible
    assert event.eligibility_reason == "backdated_or_future_recommendation_date"
    assert registry.current_active_state() == ()


def test_live_registration_uses_new_york_market_date_after_midnight_utc(tmp_path, monkeypatch):
    after_midnight_utc = dt.datetime(2026, 7, 11, 0, 30, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(forward_registry, "_utc_now", lambda: after_midnight_utc)
    registry = _registry(tmp_path)

    market_date_event = _register(
        registry,
        recommendation_date=dt.date(2026, 7, 10),
    )
    utc_date_event = _register(
        registry,
        logical_id="utc-date-event",
        recommendation_date=dt.date(2026, 7, 11),
    )

    assert market_date_event.eligible
    assert market_date_event.recommendation_date == dt.date(2026, 7, 10)
    assert market_date_event.registered_at == after_midnight_utc
    assert not utc_date_event.eligible
    assert utc_date_event.eligibility_reason == "backdated_or_future_recommendation_date"


def test_weekend_registration_accepts_bounded_latest_market_session(tmp_path, monkeypatch):
    saturday = dt.datetime(2026, 7, 11, 16, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(forward_registry, "_utc_now", lambda: saturday)
    registry = _registry(tmp_path)

    event = _register(
        registry,
        recommendation_date=dt.date(2026, 7, 10),
        live_market_session_date=dt.date(2026, 7, 10),
    )

    assert event.eligible
    assert event.eligibility_reason == "live_market_session_date"
    assert event.run_provenance["live_market_session_date"] == "2026-07-10"


def test_live_market_session_override_rejects_arbitrary_backdating(tmp_path, monkeypatch):
    saturday = dt.datetime(2026, 7, 11, 16, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(forward_registry, "_utc_now", lambda: saturday)
    registry = _registry(tmp_path)

    with pytest.raises(forward_registry.RegistryValidationError, match="prior 7 calendar days"):
        _register(
            registry,
            recommendation_date=dt.date(2026, 7, 1),
            live_market_session_date=dt.date(2026, 7, 1),
        )


def test_shadow_recommendation_excludes_review_rows(tmp_path):
    registry = _registry(tmp_path)
    _register(
        registry,
        status="REVIEW",
        code_provenance={"git_commit": "abc123"},
        run_provenance={
            "ticker": "SPY",
            "trade_plan": "BUY 1 SPY 2026-07-17 630 Call / SELL 1 SPY 2026-07-17 635 Call @ 2.00 DEBIT",
            "strategy_route": "bull_call_spread",
            "entry_type": "DEBIT",
            "entry_limit": 2.0,
            "target_entry": 1.9,
            "target_exit": 3.2,
            "expiry": "2026-07-17",
        },
    )

    shadow = core.build_prospective_shadow_recommendations(registry.path)

    assert shadow.empty


def test_shadow_recommendation_excludes_non_green_registration(tmp_path, monkeypatch):
    saturday = dt.datetime(2026, 7, 11, 16, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(forward_registry, "_utc_now", lambda: saturday)
    registry = _registry(tmp_path)
    _register(
        registry,
        recommendation_date=dt.date(2026, 7, 11),
        status="REVIEW",
    )

    shadow = core.build_prospective_shadow_recommendations(registry.path)

    assert shadow.empty


def test_shadow_expectancy_selection_honors_promoted_daily_cap(tmp_path):
    registry = _registry(tmp_path)
    daily_cap = core._promoted_selector_daily_cap()
    registrations = [("first-t000", "T000"), ("duplicate-t000", "T000")]
    registrations.extend(
        (f"unique-{idx:03d}", f"T{idx:03d}")
        for idx in range(1, daily_cap + 2)
    )
    for logical_id, ticker in registrations:
        _register(
            registry,
            logical_id=logical_id,
            status="GREEN",
            run_provenance={
                "ticker": ticker,
                "strategy_route": "bull_call_debit",
            },
        )

    shadow = core.build_prospective_shadow_recommendations(registry.path)
    selected = shadow[shadow["selected_for_expectancy"].map(bool)].sort_values(
        "evidence_selection_rank"
    )

    assert selected["ticker"].tolist() == [f"T{idx:03d}" for idx in range(daily_cap)]
    assert len(selected) == daily_cap
    assert selected["evidence_selection_rank"].tolist() == list(range(1, daily_cap + 1))
    assert set(shadow["evidence_selection_policy"]) == {
        f"top_{daily_cap}_entry_proven_unique_tickers_by_frozen_rank_then_sequence_v3"
    }
    assert not shadow["execution_permission"].map(bool).any()


def test_shadow_expectancy_selection_excludes_non_green_rows(tmp_path):
    registry = _registry(tmp_path)
    _register(
        registry,
        logical_id="gray-review",
        status="REVIEW",
        run_provenance={"ticker": "SPY", "strategy_route": "bull_call_debit"},
    )
    _register(
        registry,
        logical_id="yellow-target",
        status="GREEN",
        run_provenance={
            "ticker": "QQQ",
            "strategy_route": "bull_call_debit",
        },
    )

    shadow = core.build_prospective_shadow_recommendations(registry.path)

    selected = shadow[shadow["selected_for_expectancy"].map(bool)]
    assert selected["ticker"].tolist() == ["QQQ"]
    assert shadow["ticker"].tolist() == ["QQQ"]


def test_exact_later_fill_matches_one_active_recommendation_and_normalizes_ratio(tmp_path):
    registry = _registry(tmp_path)
    recommendation = _register(registry)
    two_lot_fill = [
        {"side": "BUY", "qty": 2, "occ_symbol": BUY_LEG},
        {"side": "SELL", "qty": 2, "occ_symbol": SELL_LEG},
    ]

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp=NOW,
        legs=two_lot_fill,
    )

    assert match.matched
    assert match.reason == BrokerMatchReason.MATCHED
    assert match.recommendation == recommendation


def test_schwab_basic_utc_offset_fill_timestamp_is_accepted(tmp_path):
    registry = _registry(tmp_path)
    recommendation = _register(registry)

    match = registry.match_broker_fill(
        account_id="acct-123",
        fill_timestamp="2026-07-10T18:30:01+0000",
        legs=_legs(),
    )

    assert match.matched
    assert match.reason == BrokerMatchReason.MATCHED
    assert match.recommendation == recommendation


def test_core_registers_only_final_live_directed_tickets(tmp_path):
    tickets = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "trade_plan": "BUY 1 SPY 2026-07-17 630 Call / SELL 1 SPY 2026-07-17 635 Call @ 2.00 DEBIT",
                "entry_limit": 2.0,
                "target_entry": 2.0,
                "live_validation_status": "PASS",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
            }
        ]
    )
    path = tmp_path / "registry.jsonl"

    skipped = core.register_prospective_options_agent_recommendations(
        tickets,
        root=tmp_path,
        out_dir=tmp_path / "snapshot",
        source_date="2026-07-09",
        live_schwab=False,
        live_portfolio=True,
        chain_snapshot_dir=tmp_path / "chains",
        agent_reviews_json=tmp_path / "reviews.json",
        registry_path=path,
        recommendation_date=NOW.date(),
    )
    assert skipped["status"] == "skipped_not_final_live_run"
    assert not path.exists()

    registered = core.register_prospective_options_agent_recommendations(
        tickets,
        root=tmp_path,
        out_dir=tmp_path / "live",
        source_date="2026-07-09",
        live_schwab=True,
        live_portfolio=True,
        chain_snapshot_dir=None,
        agent_reviews_json=tmp_path / "reviews.json",
        registry_path=path,
        recommendation_date=NOW.date(),
    )

    assert registered["status"] == "registered"
    assert registered["registered_events"] == 1
    state = ForwardRecommendationRegistry(path).current_active_state(account_id="acct_3326")
    assert len(state) == 1
    assert [(leg.side, leg.occ_symbol) for leg in state[0].legs] == [
        ("BUY", BUY_LEG),
        ("SELL", SELL_LEG),
    ]


def test_core_does_not_register_non_green_ticket_surfaces(tmp_path):
    tickets = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "trade_plan": "BUY 1 SPY 2026-07-17 630 Call / SELL 1 SPY 2026-07-17 635 Call @ 2.00 DEBIT",
                "entry_limit": 2.0,
                "live_validation_status": "PASS",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
            },
            {
                "ticker": "QQQ",
                "trade_plan": "BUY 1 QQQ 2026-07-17 560 Call / SELL 1 QQQ 2026-07-17 565 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
            },
        ]
    )
    path = tmp_path / "registry.jsonl"

    summary = core.register_prospective_options_agent_recommendations(
        tickets,
        root=tmp_path,
        out_dir=tmp_path / "live",
        source_date="2026-07-09",
        live_schwab=True,
        live_portfolio=True,
        chain_snapshot_dir=None,
        agent_reviews_json=tmp_path / "reviews.json",
        registry_path=path,
        recommendation_date=NOW.date(),
    )

    registry = ForwardRecommendationRegistry(path)
    assert summary["registered_events"] == 1
    assert len(registry.events()) == 1
    assert len(registry.current_active_state(account_id="acct_3326")) == 1
