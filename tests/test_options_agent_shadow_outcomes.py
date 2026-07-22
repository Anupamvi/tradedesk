import datetime as dt
import json

import pandas as pd

from uwos.options_agent import core
from uwos.options_agent.shadow_outcomes import collect_due_shadow_outcomes


SESSION = dt.date(2026, 7, 17)
QUOTE_TIME = dt.datetime(2026, 7, 17, 20, 0, tzinfo=dt.timezone.utc)


def _shadow(*, logical_id="oa-shadow-1", entry_type="DEBIT", entry_limit=2.0, due="2026-07-17"):
    return pd.DataFrame(
        [
            {
                "logical_recommendation_id": logical_id,
                "recommendation_date": "2026-07-10",
                "evaluation_due_date": due,
                "registration_status": "VALID_PROSPECTIVE",
                "ticker": "SPY",
                "strategy_route": "bull_call_debit",
                "strategy_family": "long_call",
                "entry_type": entry_type,
                "direction_bucket": "bullish",
                "regime": "risk_on",
                "dte": 14,
                "dte_bucket": "dte_0_14",
                "iv_rank_bucket": "iv_mid",
                "economics_bucket": "debit_rr_mid",
                "liquidity_bucket": "liquidity_deep",
                "selected_for_expectancy": True,
                "pipeline_version": core.PIPELINE_VERSION,
                "selector_policy_id": core.PROMOTED_SELECTOR_POLICY_ID,
                "entry_limit": entry_limit,
                "expiry": "2026-07-24",
                "trade_plan": "BUY SPY call / SELL SPY call",
                "code_git_sha": "abc123",
                "legs_json": json.dumps(
                    [
                        {"side": "BUY", "ratio": 1, "occ_symbol": "SPY260724C00630000"},
                        {"side": "SELL", "ratio": 1, "occ_symbol": "SPY260724C00635000"},
                    ]
                ),
            }
        ]
    )


def _quotes(*, quote_time=QUOTE_TIME):
    millis = int(quote_time.timestamp() * 1000)
    return {
        "SPY   260724C00630000": {
            "quote": {"bidPrice": 3.0, "askPrice": 3.2, "quoteTimeInLong": millis}
        },
        "SPY   260724C00635000": {
            "quote": {"bidPrice": 1.0, "askPrice": 1.2, "quoteTimeInLong": millis}
        },
    }


def test_debit_shadow_uses_conservative_liquidation_and_appends_once(tmp_path):
    registry = tmp_path / "shadow_outcomes.jsonl"
    fetch_calls = []

    def fetch(symbols):
        fetch_calls.append(tuple(symbols))
        return _quotes()

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=registry,
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=fetch,
        observed_at=QUOTE_TIME,
    )
    repeated, repeated_attempts, repeated_summary = collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=registry,
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: (_ for _ in ()).throw(AssertionError("must not refetch")),
        observed_at=QUOTE_TIME,
    )

    assert len(fetch_calls) == 1
    assert len(outcomes) == len(repeated) == 1
    assert outcomes.iloc[0]["liquidation_value"] == 1.8
    assert outcomes.iloc[0]["realized_pnl"] == -20.0
    assert bool(outcomes.iloc[0]["exact_evaluated"])
    assert bool(outcomes.iloc[0]["contributes_to_expectancy"])
    assert outcomes.iloc[0]["selector_policy_id"] == core.PROMOTED_SELECTOR_POLICY_ID
    assert outcomes.iloc[0]["recommendation_pipeline_version"] == core.PIPELINE_VERSION
    assert attempts.iloc[0]["status"] == "SCORED"
    assert repeated_attempts.iloc[0]["status"] == "ALREADY_SCORED"
    assert summary["new_outcome_rows"] == 1
    assert repeated_summary["new_outcome_rows"] == 0
    assert len(registry.read_text().splitlines()) == 1


def test_credit_shadow_uses_ask_to_close_short_and_bid_to_sell_long(tmp_path):
    shadow = _shadow(entry_type="CREDIT", entry_limit=1.3)
    shadow.at[0, "strategy_route"] = "bear_call_credit"
    shadow.at[0, "legs_json"] = json.dumps(
        [
            {"side": "SELL", "ratio": 1, "occ_symbol": "SPY260724C00630000"},
            {"side": "BUY", "ratio": 1, "occ_symbol": "SPY260724C00635000"},
        ]
    )

    outcomes, attempts, _ = collect_due_shadow_outcomes(
        shadow,
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: _quotes(),
        observed_at=QUOTE_TIME,
    )

    assert attempts.iloc[0]["status"] == "SCORED"
    assert outcomes.iloc[0]["liquidation_value"] == -2.2
    assert outcomes.iloc[0]["realized_pnl"] == -90.0


def test_stale_quote_blocks_scoring_and_does_not_append(tmp_path):
    stale = dt.datetime(2026, 7, 16, 19, 0, tzinfo=dt.timezone.utc)
    registry = tmp_path / "shadow_outcomes.jsonl"

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=registry,
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: _quotes(quote_time=stale),
        observed_at=QUOTE_TIME,
    )

    assert outcomes.empty
    assert attempts.iloc[0]["status"] == "BLOCKED"
    assert "quote_not_from_evaluation_session" in attempts.iloc[0]["blocker"]
    assert summary["contributing_rows"] == 0
    assert not registry.exists()


def test_shadow_outcome_accepts_schwab_quote_time_field(tmp_path):
    quotes = _quotes()
    for payload in quotes.values():
        quote = payload["quote"]
        quote["quoteTime"] = quote.pop("quoteTimeInLong")

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: quotes,
        observed_at=QUOTE_TIME,
    )

    assert attempts.iloc[0]["status"] == "SCORED"
    assert len(outcomes) == 1
    assert summary["new_outcome_rows"] == 1


def test_due_shadow_waits_until_near_close_before_fetching(tmp_path):
    early = dt.datetime(2026, 7, 17, 14, 0, tzinfo=dt.timezone.utc)

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: (_ for _ in ()).throw(AssertionError("must not fetch early")),
        observed_at=early,
    )

    assert outcomes.empty
    assert attempts.iloc[0]["status"] == "NOT_DUE"
    assert attempts.iloc[0]["blocker"] == "wait_until_near_close_for_same_session_quotes"
    assert summary["due_rows"] == 0


def test_missed_fixed_session_is_audited_but_never_scored(tmp_path):
    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(due="2026-07-16"),
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: (_ for _ in ()).throw(AssertionError("must not fetch")),
        observed_at=QUOTE_TIME,
    )

    assert outcomes.empty
    assert attempts.iloc[0]["status"] == "MISSED"
    assert attempts.iloc[0]["blocker"] == "fixed_evaluation_session_missed"
    assert summary["missed_rows"] == 1


def test_missed_fixed_session_is_recovered_from_dated_exact_quotes(tmp_path):
    due_day = dt.date(2026, 7, 16)
    due_quote_time = dt.datetime(2026, 7, 16, 20, 0, tzinfo=dt.timezone.utc)
    fetches = []

    def historical_fetch(day, symbols):
        fetches.append((day, tuple(symbols)))
        return _quotes(quote_time=due_quote_time)

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        _shadow(due=due_day.isoformat()),
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=False,
        historical_quote_fetcher=historical_fetch,
        observed_at=QUOTE_TIME,
    )

    assert fetches == [
        (
            due_day,
            ("SPY260724C00630000", "SPY260724C00635000"),
        )
    ]
    assert attempts.iloc[0]["status"] == "SCORED"
    assert attempts.iloc[0]["observation_session_date"] == due_day.isoformat()
    assert outcomes.iloc[0]["observation_session_date"] == due_day.isoformat()
    assert outcomes.iloc[0]["quote_source"] == "dated_uw_exact_option_quotes"
    assert summary["missed_rows"] == 0
    assert summary["new_outcome_rows"] == 1


def test_nonselected_exact_shadow_is_diagnostic_and_not_expectancy(tmp_path):
    shadow = _shadow()
    shadow.at[0, "selected_for_expectancy"] = False

    outcomes, attempts, summary = collect_due_shadow_outcomes(
        shadow,
        outcome_registry_path=tmp_path / "shadow_outcomes.jsonl",
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: _quotes(),
        observed_at=QUOTE_TIME,
    )

    assert attempts.iloc[0]["status"] == "SCORED"
    assert bool(outcomes.iloc[0]["exact_evaluated"])
    assert not bool(outcomes.iloc[0]["contributes_to_expectancy"])
    assert outcomes.iloc[0]["outcome_status"] == "SCORED_EXACT_FIXED_HORIZON_DIAGNOSTIC"
    assert summary["contributing_rows"] == 0


def test_actual_evidence_excludes_shadow_outcomes_from_obsolete_selector() -> None:
    outcomes = pd.DataFrame(
        [
            {"selector_policy_id": core.PROMOTED_SELECTOR_POLICY_ID, "realized_pnl": 100.0},
            {"selector_policy_id": "obsolete_selector", "realized_pnl": 10_000.0},
        ]
    )

    active = core._active_selector_shadow_outcomes(outcomes)

    assert active["realized_pnl"].tolist() == [100.0]


def test_scored_shadow_enters_actual_calibration_and_expectancy_once(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    registry = out_dir / "options_agent_shadow_outcomes.jsonl"
    collect_due_shadow_outcomes(
        _shadow(),
        outcome_registry_path=registry,
        observation_session_date=SESSION,
        live_schwab=True,
        quote_fetcher=lambda symbols: _quotes(),
        observed_at=QUOTE_TIME,
    )

    actual = core._actual_calibration_frame(tmp_path, out_dir)
    expectancy = core.build_expectancy_evidence(
        tmp_path,
        pd.DataFrame([{"ticker": "SPY", "strategy_route": "bull_call_debit"}]),
        pd.DataFrame(),
    )

    shadow_actual = actual[actual["source"].astype(str).eq("options_agent_shadow_outcomes")]
    assert len(shadow_actual) == 1
    row = shadow_actual.iloc[0]
    assert row["entry_order_ids"] == "oa-shadow-1"
    assert row["strategy_route"] == "bull_call_debit"
    assert row["regime"] == "risk_on"
    assert row["liquidity_bucket"] == "liquidity_deep"
    evidence = expectancy[expectancy["source"].astype(str).eq("options_agent_shadow_outcomes")]
    assert len(evidence) == 1
    assert evidence.iloc[0]["sample_size"] == 1
    assert evidence.iloc[0]["avg_pnl"] == -20.0
