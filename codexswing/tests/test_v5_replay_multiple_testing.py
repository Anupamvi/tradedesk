import pytest

from codexswing.backtest.labels import DailyBar
from codexswing.v5.exact_path_replay import replay_exact_option_path
from codexswing.v5.multiple_testing import (
    HypothesisEvidence,
    cluster_bootstrap_one_sided_pvalue,
    evaluate_hypothesis_family,
    holm_bonferroni,
)
from codexswing.v5.replay_plan import (
    ReplayPathSample,
    SessionPnL,
    build_replay_paths,
    cache_requirements_for_paths,
    choose_path_exit,
    declared_variants,
)
from codexswing.v5.spec import V5ResearchSpec


def _spec(project_root):
    return V5ResearchSpec.from_json_file(
        project_root / "research_specs" / "ORATS_SWING_RESEARCH_V5.json"
    )


def _bars(count=25):
    return tuple(
        DailyBar(
            ticker="SPY",
            trade_date="2026-01-{:02d}".format(index),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
        )
        for index in range(1, count + 1)
    )


def test_variants_and_path_requirements_are_predeclared(project_root):
    spec = _spec(project_root)
    variants = declared_variants(spec)
    assert len(variants) == 72
    assert len({item.hypothesis_id for item in variants}) == 72

    paths = build_replay_paths("SPY", ["2026-01-01"], _bars(), spec.horizons_sessions)
    assert [item.horizon_sessions for item in paths] == [3, 5, 10, 20]
    requirements = cache_requirements_for_paths(paths)
    strike_dates = [item.session_date for item in requirements if item.endpoint == "hist/strikes"]
    assert len(strike_dates) == 20
    assert strike_dates[0] == "2026-01-02"
    assert strike_dates[-1] == "2026-01-21"


def test_dynamic_exit_uses_first_fixed_target_or_stop(project_root):
    spec = _spec(project_root)
    target_policy = next(item for item in spec.exit_policies if item.policy_id == "PT_025R_SL_035R")
    target = choose_path_exit(
        [
            SessionPnL("2026-01-02", 10.0),
            SessionPnL("2026-01-03", 26.0),
            SessionPnL("2026-01-04", -50.0),
        ],
        maximum_risk_dollars=100.0,
        exit_policy=target_policy,
    )
    assert target.reason == "PROFIT_TARGET"
    assert target.session_number == 2

    stop = choose_path_exit(
        [SessionPnL("2026-01-02", -36.0), SessionPnL("2026-01-03", 50.0)],
        maximum_risk_dollars=100.0,
        exit_policy=target_policy,
    )
    assert stop.reason == "STOP_LOSS"
    assert stop.session_number == 1


def test_exact_single_option_path_closes_on_predeclared_target(project_root):
    spec = _spec(project_root)
    policy = next(item for item in spec.exit_policies if item.policy_id == "PT_025R_SL_035R")
    sample = ReplayPathSample(
        ticker="SPY",
        decision_date="2026-01-01",
        entry_date="2026-01-02",
        path_dates=("2026-01-02", "2026-01-03", "2026-01-04"),
        horizon_sessions=3,
    )

    def row(trade_date, bid, ask):
        return {
            "ticker": "SPY",
            "tradeDate": trade_date,
            "expirDate": "2026-02-06",
            "strike": 100,
            "stockPrice": 100,
            "callBidPrice": bid,
            "callAskPrice": ask,
            "callVolume": 50,
            "callOpenInterest": 500,
            "delta": 0.52,
        }

    result = replay_exact_option_path(
        sample,
        "LONG_CALL",
        policy,
        {
            "2026-01-02": [row("2026-01-02", 4.0, 4.4)],
            "2026-01-03": [row("2026-01-03", 5.5, 5.7)],
        },
    )

    assert result.disposition == "CLOSED"
    assert result.exit_decision.reason == "PROFIT_TARGET"
    assert result.exit_decision.session_date == "2026-01-03"
    assert len(result.pnl_path) == 2


def test_holm_is_monotone_and_cluster_bootstrap_detects_strong_positive_mean():
    adjusted = holm_bonferroni({"A": 0.01, "B": 0.03, "C": 0.04})
    assert adjusted == {
        "A": pytest.approx(0.03),
        "B": pytest.approx(0.06),
        "C": pytest.approx(0.06),
    }

    p_value = cluster_bootstrap_one_sided_pvalue(
        [10.0] * 20,
        ["cluster-{}".format(index) for index in range(20)],
        iterations=500,
    )
    assert p_value < 0.01


def test_family_gate_requires_every_attempt_and_positive_all_periods():
    good = HypothesisEvidence("GOOD", 0.001, 2.0, 1.0, 1.0, 20)
    bad = HypothesisEvidence("BAD", 0.90, 2.0, -1.0, 1.0, 20)
    evaluation = evaluate_hypothesis_family(
        [good, bad], expected_hypothesis_ids={"GOOD", "BAD"}
    )
    decisions = {item.hypothesis_id: item for item in evaluation.decisions}
    assert decisions["GOOD"].status == "PROMOTION_ELIGIBLE"
    assert decisions["BAD"].status == "EXPLORATORY_NOT_PROMOTED"
    assert "VALIDATION_EXPECTANCY_NOT_POSITIVE" in decisions["BAD"].reasons

    with pytest.raises(ValueError, match="incomplete hypothesis family"):
        evaluate_hypothesis_family([good], expected_hypothesis_ids={"GOOD", "BAD"})
