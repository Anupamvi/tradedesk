from codexuw import engine
from codexuw import opportunity
from codexuw import daily_v4
from codexuw.pipeline_versions import PIPELINE_VERSION_LOCKS, PREVIOUS_PIPELINE_VERSION_LOCKS, pipeline_version_record


def test_codex_daily_pipeline_versions_are_locked() -> None:
    expected = {
        "v2": ("Codex Daily V2", "v2.2-no-signal-caps-20260826"),
        "v3": ("Codex Daily V3", "v3.2-profit-integrity-20260719"),
        "v4": ("Codex Daily V4", "v4.27-debit-production-handoff-20260816"),
    }

    for key, (name, version) in expected.items():
        record = pipeline_version_record(key)
        assert record["pipeline_name"] == name
        assert record["pipeline_version"] == version
        assert record["lock_status"] == "locked"
        expected_lock = {"v2": "2026-08-26", "v3": "2026-07-19", "v4": "2026-08-16"}[key]
        assert record["locked_on"] == expected_lock


def test_previous_codex_daily_versions_are_retained() -> None:
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v3.0"]["superseded_by"] == "v3.1-exec-confidence-20260612-143405"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v3.1-exec-confidence-20260612-143405"]["superseded_by"] == "v3.2-profit-integrity-20260719"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.0"]["superseded_by"] == "v4.1-exec-confidence-20260612-143405"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.1-exec-confidence-20260612-143405"]["superseded_by"] == "v4.2-integrity-20260713"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.2-integrity-20260713"]["superseded_by"] == "v4.3-expectancy-safe-entry-20260716"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.3-expectancy-safe-entry-20260716"]["superseded_by"] == "v4.4-confidence-calibrated-20260719"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.4-confidence-calibrated-20260719"]["superseded_by"] == "v4.5-medium-debit-sleeve-20260719"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.5-medium-debit-sleeve-20260719"]["superseded_by"] == "v4.6-walk-forward-confidence-capacity-20260719"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.6-walk-forward-confidence-capacity-20260719"]["superseded_by"] == "v4.7-policy-base-confidence-20260719"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.8.2-snapshot-validation-integrity-20260720"]["superseded_by"] == "v4.9-structure-aware-payoff-20260720"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.9-structure-aware-payoff-20260720"]["superseded_by"] == "v4.10-correlation-aware-credit-book-20260721"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.10-correlation-aware-credit-book-20260721"]["superseded_by"] == "v4.11-five-source-point-in-time-integrity-20260722"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.11-five-source-point-in-time-integrity-20260722"]["superseded_by"] == "v4.12-goal-shadow-prospective-20260723"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.12-goal-shadow-prospective-20260723"]["superseded_by"] == "v4.13-all-strategy-live-construction-20260803"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.13-all-strategy-live-construction-20260803"]["superseded_by"] == "v4.14-maturity-safe-advisory-risk-20260808"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.14-maturity-safe-advisory-risk-20260808"]["superseded_by"] == "v4.15-execution-reachability-20260810"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.15-execution-reachability-20260810"]["superseded_by"] == "v4.16-symbol-regime-credit-resolution-20260811"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.16-symbol-regime-credit-resolution-20260811"]["superseded_by"] == "v4.17-symbol-credit-exact-population-validation-20260811"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.17-symbol-credit-exact-population-validation-20260811"]["superseded_by"] == "v4.18-maturity-safe-credit-payoff-book-20260812"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.18-maturity-safe-credit-payoff-book-20260812"]["superseded_by"] == "v4.19-family-confidence-regime-capacity-20260812"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.19-family-confidence-regime-capacity-20260812"]["superseded_by"] == "v4.20-next-session-debit-route-20260812"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.20-next-session-debit-route-20260812"]["superseded_by"] == "v4.21-directional-credit-medium-book-20260812"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.21-directional-credit-medium-book-20260812"]["superseded_by"] == "v4.22-execution-parity-no-signal-caps-20260813"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.22-execution-parity-no-signal-caps-20260813"]["superseded_by"] == "v4.23-corrected-fill-stress-execution-population-20260813"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.23-corrected-fill-stress-execution-population-20260813"]["superseded_by"] == "v4.24-effective-payoff-evidence-precedence-20260813"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.24-effective-payoff-evidence-precedence-20260813"]["superseded_by"] == "v4.25-actionable-ticket-contract-dedupe-20260814"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.25-actionable-ticket-contract-dedupe-20260814"]["superseded_by"] == "v4.26-bull-call-debit-pilot-readable-board-20260816"
    assert PREVIOUS_PIPELINE_VERSION_LOCKS["v4.26-bull-call-debit-pilot-readable-board-20260816"]["superseded_by"] == "v4.27-debit-production-handoff-20260816"


def test_locked_versions_mark_schwab_live_portfolio_and_gex_context() -> None:
    for record in PIPELINE_VERSION_LOCKS.values():
        assert record["live_schwab_required_for_execute"] is True
        assert record["portfolio_state_required_for_execute"] is True
        assert "gex" in record["gex_context"].lower() or "gamma" in record["gex_context"].lower()


def test_pipeline_constants_match_locked_registry() -> None:
    assert engine.PIPELINE_NAME == "Codex Daily V2"
    assert engine.PIPELINE_VERSION == "v2.2-no-signal-caps-20260826"
    assert opportunity.PIPELINE_NAME_V3 == "Codex Daily V3"
    assert opportunity.PIPELINE_VERSION_V3 == "v3.2-profit-integrity-20260719"
    assert daily_v4.PIPELINE_NAME_V4 == "Codex Daily V4"
    assert daily_v4.PIPELINE_VERSION_V4 == "v4.27-debit-production-handoff-20260816"
