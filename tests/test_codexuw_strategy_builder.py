from __future__ import annotations

import datetime as dt

import pandas as pd

import codexuw.engine as engine
from codexuw.strategy_registry import STRATEGY_SPECS
from codexuw.schwab_live import is_regular_option_session_open
from codexuw.strategy_builder import (
    GENERIC_STRATEGY_SPECS,
    build_generic_strategy_candidate,
    historical_scope_for_strategy,
)


ASOF = dt.date(2026, 8, 3)


def _chain(*, displayed_size: int = 20, regular_session: bool = True) -> pd.DataFrame:
    rows = []
    for expiry in (dt.date(2026, 10, 16), dt.date(2026, 11, 13)):
        for right in ("C", "P"):
            for strike in range(80, 121, 5):
                mid = max(0.50, 8.0 - abs(strike - 100) * 0.25)
                rows.append(
                    {
                        "expiry": expiry,
                        "right": right,
                        "strike": float(strike),
                        "symbol": f"X-{expiry}-{right}-{strike}",
                        "bid": mid - 0.05,
                        "ask": mid + 0.05,
                        "mark": mid,
                        "bid_size": displayed_size,
                        "ask_size": displayed_size,
                        "open_interest": 1_000,
                        "volume": 100,
                        "regular_session_quote": regular_session,
                    }
                )
    return pd.DataFrame(rows)


def test_every_non_vertical_registry_family_has_a_live_generic_constructor() -> None:
    chain = _chain()

    results = {
        spec.key: build_generic_strategy_candidate(
            chain,
            strategy_key=spec.key,
            spot=100.0,
            as_of_date=ASOF,
        )
        for spec in GENERIC_STRATEGY_SPECS
    }

    assert len(results) == 28
    assert all(result["live_status"] == "PASS" for result in results.values())
    assert results["long_call"]["leg_count"] == 1
    assert results["long_straddle"]["leg_count"] == 2
    assert results["call_butterfly"]["leg_count"] == 3
    assert results["iron_condor"]["leg_count"] == 4
    assert results["covered_call"]["leg_count"] == 2
    assert results["iron_condor"]["short_oi"] == 1_000
    assert results["iron_condor"]["long_oi"] == 1_000
    assert all(historical_scope_for_strategy(key) != "unavailable" for key in results)


def test_calendar_uses_two_expiries_and_natural_side_sizes() -> None:
    result = build_generic_strategy_candidate(
        _chain(displayed_size=12),
        strategy_key="call_calendar",
        spot=100.0,
        as_of_date=ASOF,
    )

    assert result["live_status"] == "PASS"
    assert result["expiry"] == dt.date(2026, 10, 16)
    assert result["far_expiry"] == dt.date(2026, 11, 13)
    assert result["displayed_entry_size"] == 12
    assert result["path_dependent_structure"]


def test_ratio_quantity_reduces_executable_size() -> None:
    result = build_generic_strategy_candidate(
        _chain(displayed_size=9),
        strategy_key="call_ratio_spread",
        spot=100.0,
        as_of_date=ASOF,
    )

    assert result["live_status"] == "PASS"
    assert result["displayed_entry_size"] == 4
    assert result["requires_margin_model"]


def test_stock_backed_prices_option_overlay_but_keeps_full_payoff_risk() -> None:
    covered_call = build_generic_strategy_candidate(
        _chain(),
        strategy_key="covered_call",
        spot=100.0,
        as_of_date=ASOF,
    )
    protective_put = build_generic_strategy_candidate(
        _chain(),
        strategy_key="protective_put",
        spot=100.0,
        as_of_date=ASOF,
    )

    assert covered_call["entry_type"] == "credit"
    assert covered_call["entry_price"] < 10
    assert covered_call["requires_equity_shares"] == 100
    assert covered_call["max_profit"] > 0
    assert covered_call["max_loss"] > 0
    assert protective_put["entry_type"] == "debit"
    assert protective_put["max_loss"] > 0


def test_zero_size_or_preopen_chain_stays_visible_but_non_executable() -> None:
    result = build_generic_strategy_candidate(
        _chain(displayed_size=0, regular_session=False),
        strategy_key="iron_condor",
        spot=100.0,
        as_of_date=ASOF,
    )

    assert result["live_status"] == "PASS"
    assert result["displayed_entry_size"] == 0
    assert not result["regular_session_quote"]
    assert result["execution_authority"] == "research_only_pending_strategy_validation"


def test_discovery_emits_all_32_families_for_one_ticker() -> None:
    stock = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "Technology",
                "close": 100.0,
                "flow_bias": 0.10,
                "flow_total_premium": 200_000_000,
                "iv30d": 0.30,
                "iv_rank": 45.0,
                "realized_volatility_30d": 0.22,
            }
        ]
    )
    hot = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "dte": 30,
                "expiry_dt": dt.date(2026, 9, 2),
                "right": right,
                "strike": float(strike),
                "premium": 1_000_000,
                "volume": 1_000,
                "open_interest": 2_000,
                "bid": 2.0,
                "ask": 2.1,
                "option_symbol": f"AAA-{right}-{strike}",
            }
            for right in ("C", "P")
            for strike in (85, 90, 95, 100, 105, 110, 115)
        ]
    )

    candidates = engine.generate_candidates(
        stock,
        hot,
        pd.DataFrame(),
        asof=ASOF,
        max_candidates=0,
    )

    expected = {str(spec["strategy_key"]) for spec in STRATEGY_SPECS}
    assert set(candidates["strategy_registry_key"]) == expected


def test_live_validation_constructs_all_generic_families(monkeypatch, tmp_path) -> None:
    class FakeValidator:
        errors = {}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_chain(self, *args, **kwargs) -> dict:
            return {"status": "SUCCESS"}

        def save(self) -> None:
            pass

    monkeypatch.setattr(engine, "SchwabChainValidator", FakeValidator)
    monkeypatch.setattr(engine, "chain_spot", lambda chain: 100.0)
    monkeypatch.setattr(engine, "chain_to_contracts", lambda chain: _chain())
    seeds = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "strategy_key": spec.key,
                "strategy_registry_key": spec.key,
                "strategy": spec.display_name,
                "direction": spec.direction,
                "strategy_kind": spec.strategy_kind,
                "generic_strategy_seed": True,
                "expiry": dt.date(2026, 9, 18),
                "dte": 46,
                "combined_flow_bias": 0.0,
                "flow_total_premium": 200_000_000,
                "iv_rank": 45.0,
                "iv30d": 0.30,
                "realized_volatility_30d": 0.22,
            }
            for spec in GENERIC_STRATEGY_SPECS
        ]
    )

    scored = engine.live_validate_and_score(
        seeds,
        asof=ASOF,
        out_dir=tmp_path,
        regime={"trend": "uptrend", "transition": False},
        require_live=True,
    )

    assert scored["strategy_registry_key"].nunique() == 28
    assert scored["live_status"].eq("PASS").all()
    assert scored["replay_ev_verdict"].eq("unavailable_generic_strategy").all()


def test_nan_generic_seed_does_not_reroute_vertical(monkeypatch, tmp_path) -> None:
    class FakeValidator:
        errors = {}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_chain(self, *args, **kwargs) -> dict:
            return {"status": "SUCCESS"}

        def save(self) -> None:
            pass

    monkeypatch.setattr(engine, "SchwabChainValidator", FakeValidator)
    monkeypatch.setattr(engine, "chain_spot", lambda chain: 100.0)
    monkeypatch.setattr(engine, "chain_to_contracts", lambda chain: _chain())
    monkeypatch.setattr(
        engine,
        "find_credit_spread_alternatives",
        lambda *args, **kwargs: [{"live_status": "PASS", "credit": 1.25, "natural_credit": 1.20, "mid_credit": 1.30, "spread_width": 5.0, "short_oi": 1000, "short_volume": 100, "long_oi": 1000, "long_volume": 100}],
    )
    seed = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "strategy_kind": "Credit",
                "strategy_registry_key": "bull_put_credit_vertical",
                "generic_strategy_seed": float("nan"),
                "expiry": dt.date(2026, 9, 18),
                "dte": 46,
                "combined_flow_bias": 0.10,
                "flow_total_premium": 200_000_000,
            }
        ]
    )

    scored = engine.live_validate_and_score(
        seed,
        asof=ASOF,
        out_dir=tmp_path,
        regime={"trend": "uptrend", "transition": False},
        require_live=True,
    )

    assert scored.iloc[0]["live_status"] == "PASS"
    assert scored.iloc[0]["strategy_registry_key"] == "bull_put_credit_vertical"


def test_stale_prior_session_leg_is_not_executable_against_newer_chain_date(monkeypatch, tmp_path) -> None:
    class FakeValidator:
        errors = {}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_chain(self, *args, **kwargs) -> dict:
            return {"status": "SUCCESS"}

        def save(self) -> None:
            pass

    contracts = _chain()
    contracts["quote_date"] = dt.date(2026, 7, 31)
    contracts.loc[contracts.index[0], "quote_date"] = dt.date(2026, 8, 3)
    contracts.loc[contracts.index[0], "regular_session_quote"] = False
    monkeypatch.setattr(engine, "SchwabChainValidator", FakeValidator)
    monkeypatch.setattr(engine, "chain_spot", lambda chain: 100.0)
    monkeypatch.setattr(engine, "chain_to_contracts", lambda chain: contracts.copy())
    seed = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "strategy_key": "iron_condor",
                "strategy_registry_key": "iron_condor",
                "strategy": "Iron Condor",
                "direction": "Iron Condor",
                "strategy_kind": "Credit",
                "generic_strategy_seed": True,
                "expiry": dt.date(2026, 9, 18),
                "dte": 49,
                "combined_flow_bias": 0.0,
                "flow_total_premium": 200_000_000,
            }
        ]
    )

    scored = engine.live_validate_and_score(
        seed,
        asof=dt.date(2026, 7, 31),
        out_dir=tmp_path,
        regime={"trend": "uptrend", "transition": False},
        require_live=True,
    )

    assert scored.iloc[0]["quote_observation_date"] == dt.date(2026, 8, 3)
    assert not bool(scored.iloc[0]["regular_session_quote"])


def test_regular_option_session_boundaries() -> None:
    eastern = dt.timezone(dt.timedelta(hours=-4))

    assert is_regular_option_session_open(dt.datetime(2026, 8, 3, 10, 0, tzinfo=eastern))
    assert is_regular_option_session_open(dt.datetime(2026, 8, 3, 16, 0, tzinfo=eastern))
    assert not is_regular_option_session_open(dt.datetime(2026, 8, 3, 16, 1, tzinfo=eastern))
    assert not is_regular_option_session_open(dt.datetime(2026, 8, 3, 9, 0, tzinfo=eastern))
    assert not is_regular_option_session_open(dt.datetime(2026, 8, 3, 17, 0, tzinfo=eastern))
    assert not is_regular_option_session_open(dt.datetime(2026, 8, 2, 10, 0, tzinfo=eastern))


def test_snapshot_directory_disables_live_fallback(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeValidator:
        errors = {"AAA": "snapshot missing"}

        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def get_chain(self, *args, **kwargs):
            return None

        def save(self) -> None:
            pass

    monkeypatch.setattr(engine, "SchwabChainValidator", FakeValidator)
    seed = pd.DataFrame(
        [{"ticker": "AAA", "strategy_registry_key": "long_call", "generic_strategy_seed": True, "expiry": dt.date(2026, 9, 18)}]
    )

    scored = engine.live_validate_and_score(
        seed,
        asof=ASOF,
        out_dir=tmp_path / "out",
        regime={"trend": "uptrend", "transition": False},
        require_live=True,
        schwab_snapshot_dir=tmp_path / "snapshots",
    )

    assert captured["allow_live_fallback"] is False
    assert scored.iloc[0]["live_status"] == "chain_error"
