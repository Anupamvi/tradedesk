import pandas as pd

from codexuw.debit_walkforward_shadow import (
    _profit_factor_pass,
    candidate_guard,
    evaluate,
    learning_guard,
    payoff_aware_expected_value,
    select_book,
)


def _candidate_row() -> dict:
    return {
        "exact_evaluated": True,
        "entry_debit": 1.0,
        "entry_width": 5.0,
        "exit_value": 2.0,
        "asof": pd.Timestamp("2026-01-02"),
        "entry_day": pd.Timestamp("2026-01-05"),
        "exit_day": pd.Timestamp("2026-01-06"),
        "earnings_known": True,
        "earnings_crosses": False,
        "entry_dte": 30,
        "debit_pct_width": 0.20,
        "entry_quote_width_pct": 0.10,
        "reward_risk": 4.0,
        "breakeven_sigma": 0.50,
        "iv_hv_ratio": 1.00,
        "replay_guard_pass": False,
    }


def test_candidate_guard_does_not_depend_on_old_replay_policy() -> None:
    assert bool(candidate_guard(pd.DataFrame([_candidate_row()])).iloc[0])


def test_learning_guard_rejects_same_session_entry() -> None:
    row = _candidate_row()
    row["entry_day"] = row["asof"]
    assert not bool(learning_guard(pd.DataFrame([row])).iloc[0])


def test_payoff_aware_ev_uses_actual_target_not_theoretical_max_profit() -> None:
    frame = pd.DataFrame([{"entry_debit": 1.0, "entry_width": 5.0}])
    result = payoff_aware_expected_value(
        frame,
        pd.Series([0.65]),
        profit_take_pct=0.50,
        entry_stress_pct=0.10,
    )
    assert round(result.iloc[0]["conservative_win_payoff"], 2) == 40.00
    assert round(result.iloc[0]["conservative_loss_payoff"], 2) == 110.00
    assert round(result.iloc[0]["predicted_ev_payoff_correct"], 2) == -12.50


def test_all_winning_sample_passes_profit_factor_gate_without_infinity() -> None:
    assert _profit_factor_pass(
        {"n": 3, "wins": 3, "profit_factor": None},
        1.50,
    )


def test_select_book_keeps_one_highest_probability_per_day() -> None:
    frame = pd.DataFrame([
        {
            "asof": pd.Timestamp("2026-01-02"),
            "ticker": "A",
            "predicted_win_probability": 0.70,
            "predicted_ev_payoff_correct": 20.0,
            "breakeven_sigma": 0.5,
        },
        {
            "asof": pd.Timestamp("2026-01-02"),
            "ticker": "B",
            "predicted_win_probability": 0.80,
            "predicted_ev_payoff_correct": 10.0,
            "breakeven_sigma": 0.4,
        },
    ])
    selected = select_book(frame, 0.65)
    assert selected["ticker"].tolist() == ["B"]


def test_evaluate_never_grants_execution_authority_when_gates_fail() -> None:
    frame = pd.DataFrame([
        {
            "asof": pd.Timestamp("2026-04-01"),
            "strategy": "Bull Call Debit Spread",
            "predicted_win_probability": 0.70,
            "predicted_ev_payoff_correct": 20.0,
            "breakeven_sigma": 0.5,
            "stress_pnl_10pct": 50.0,
            "stress_win_10pct": 1,
        },
        {
            "asof": pd.Timestamp("2026-06-01"),
            "strategy": "Bull Call Debit Spread",
            "predicted_win_probability": 0.70,
            "predicted_ev_payoff_correct": 20.0,
            "breakeven_sigma": 0.5,
            "stress_pnl_10pct": -50.0,
            "stress_win_10pct": 0,
        },
    ])
    _, summary = evaluate(frame, cutoff="2026-05-01")
    assert summary["status"] == "RESEARCH_ONLY"
    assert summary["execution_authorized"] is False
