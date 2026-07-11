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
