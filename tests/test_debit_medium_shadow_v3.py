import datetime as dt

import pandas as pd

from codexuw import debit_medium_shadow_v3 as medium


def _row(
    ticker: str,
    strategy: str,
    probability: float,
    signal_day: str,
    pnl: float,
    payoff_ev: float = -10.0,
) -> dict:
    return {
        "ticker": ticker,
        "strategy": strategy,
        "expiry": "2026-09-18",
        "long_strike_eod": 100.0,
        "short_strike_eod": 105.0,
        "signal_day": signal_day,
        "entry_day": signal_day,
        "exit_day": "2026-05-01",
        "predicted_win_probability": probability,
        "predicted_ev_payoff_correct": payoff_ev,
        "stress_pnl_0pct": pnl + 10.0,
        "stress_pnl_5pct": pnl + 5.0,
        "stress_pnl_10pct": pnl,
        "stress_pnl_15pct": pnl - 5.0,
    }


def test_medium_lane_is_bull_only_and_does_not_require_positive_payoff_ev():
    candidates = pd.DataFrame(
        [
            _row("BULL", "Bull Call Debit Spread", 0.49, "2026-04-01", 100.0),
            _row("BEAR", "Bear Put Debit Spread", 0.80, "2026-04-02", 100.0),
            _row("LOW", "Bull Call Debit Spread", 0.44, "2026-04-03", 100.0),
        ]
    )
    selected = medium.select_medium_bull_candidates(candidates)
    assert selected["ticker"].tolist() == ["BULL"]
    assert selected.iloc[0]["predicted_ev_payoff_correct"] < 0
    assert not bool(selected.iloc[0]["execution_authorized"])
    assert selected.iloc[0]["size_multiplier"] == 0.25


def test_medium_lane_excludes_high_and_keeps_one_candidate_per_day():
    candidates = pd.DataFrame(
        [
            _row("HIGH", "Bull Call Debit Spread", 0.70, "2026-04-01", 100.0, 20.0),
            _row("NEXT", "Bull Call Debit Spread", 0.52, "2026-04-02", 90.0),
            _row("THIRD", "Bull Call Debit Spread", 0.48, "2026-04-02", 80.0),
        ]
    )
    high = candidates.iloc[[0]].copy()
    selected = medium.select_medium_bull_candidates(candidates, high_selected=high)
    assert selected["ticker"].tolist() == ["NEXT"]


def test_historical_evaluation_preserves_high_and_adds_medium():
    predictions = pd.DataFrame(
        [
            _row("HIGH", "Bull Call Debit Spread", 0.65, "2026-03-01", 100.0, 20.0),
            _row("MED", "Bull Call Debit Spread", 0.48, "2026-03-02", 80.0),
            _row("BAD", "Bear Put Debit Spread", 0.70, "2026-03-03", -100.0),
        ]
    )
    high, selected, union, summary = medium.evaluate_predictions(
        predictions, cutoff=dt.date(2026, 5, 19)
    )
    assert high["ticker"].tolist() == ["HIGH"]
    assert selected["ticker"].tolist() == ["MED"]
    assert set(union["ticker"]) == {"HIGH", "MED"}
    assert summary["execution_authorized"] is False
    assert summary["union"]["all"]["n"] == 2


def test_live_writer_emits_non_executable_artifacts(monkeypatch, tmp_path):
    candidates = pd.DataFrame(
        [_row("MED", "Bull Call Debit Spread", 0.48, "2026-08-16", 80.0)]
    )

    def fake_score(*args, **kwargs):
        return candidates, candidates.iloc[0:0].copy(), {"fake": True}

    from codexuw import daily_shadow_books

    monkeypatch.setattr(daily_shadow_books, "score_debit_shadow", fake_score)
    selected, artifacts, summary = medium.write_live_medium_outputs(
        pd.DataFrame({"ticker": ["IGNORED"]}),
        out_dir=tmp_path,
        root=tmp_path,
        asof=dt.date(2026, 8, 16),
    )
    assert selected["ticker"].tolist() == ["MED"]
    assert summary["execution_authorized"] is False
    assert all((tmp_path / path.split("/")[-1]).exists() for path in artifacts.values())
