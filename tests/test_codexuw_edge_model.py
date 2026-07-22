import pandas as pd

from codexuw.edge_model import load_replay_edge_history, match_replay_edge


def _history_row(day: str, ticker: str, *, selected: bool, distance: float, pnl: float, guard: bool = True) -> dict:
    return {
        "asof": day,
        "exit_day": "2026-01-20",
        "ticker": ticker,
        "sector": "Technology",
        "direction": "Bear Call",
        "strategy": "Bear Call Credit Spread",
        "regime": "downtrend",
        "expiry": "2026-02-06",
        "dte": 21,
        "stock_price_eod": 100.0,
        "short_strike_eod": 110.0,
        "long_strike_eod": 115.0,
        "entry_credit_pct_width": 0.25,
        "entry_quote_width_pct": 0.10,
        "expected_move_ratio": distance,
        "combined_flow_bias": -0.20,
        "flow_quality": "directional",
        "iv30d": 0.30,
        "realized_volatility_30d": 0.35,
        "exact_evaluated": True,
        "decision_pass": selected,
        "replay_guard_pass": guard,
        "pnl_1x": pnl,
        "exact_win": pnl > 0,
    }


def test_credit_edge_uses_only_selected_distance_policy_history(tmp_path) -> None:
    namespace = tmp_path / "accepted_credit_history"
    namespace.mkdir()
    rows = [
        _history_row("2026-01-05", "H1", selected=True, distance=0.80, pnl=70.0),
        _history_row("2026-01-06", "H2", selected=True, distance=0.82, pnl=65.0),
        _history_row("2026-01-07", "H3", selected=True, distance=0.85, pnl=60.0),
        _history_row("2026-01-08", "UNSELECTED", selected=False, distance=0.90, pnl=-500.0),
        _history_row("2026-01-09", "VOL_ONLY", selected=True, distance=0.50, pnl=-500.0),
    ]
    pd.DataFrame(rows).to_csv(namespace / "codexuw_replay_detail.csv", index=False)
    history = load_replay_edge_history(tmp_path, asof="2026-02-01", history_namespace="accepted_credit_history")

    edge = match_replay_edge(
        _history_row("2026-02-02", "LIVE", selected=False, distance=0.80, pnl=0.0),
        history,
    )

    assert edge["edge_sample_size"] == 3
    assert edge["edge_profit_factor"] == float("inf")
    assert edge["edge_verdict"] == "thin_sample"


def test_credit_edge_rejects_volatility_only_candidate(tmp_path) -> None:
    namespace = tmp_path / "accepted_credit_history"
    namespace.mkdir()
    pd.DataFrame(
        [_history_row("2026-01-05", "H1", selected=True, distance=0.80, pnl=70.0)]
    ).to_csv(namespace / "codexuw_replay_detail.csv", index=False)
    history = load_replay_edge_history(tmp_path, asof="2026-02-01", history_namespace="accepted_credit_history")

    edge = match_replay_edge(
        _history_row("2026-02-02", "LIVE", selected=False, distance=0.50, pnl=0.0),
        history,
    )

    assert edge["edge_match_level"] == "unavailable"
    assert "does not meet" in edge["edge_reason"]


def test_edge_history_excludes_replay_guard_failures(tmp_path) -> None:
    namespace = tmp_path / "accepted_credit_history"
    namespace.mkdir()
    rows = [
        _history_row("2026-01-05", "KEPT", selected=True, distance=0.80, pnl=70.0),
        _history_row("2026-01-06", "GUARD_FAIL", selected=True, distance=0.80, pnl=5_000.0, guard=False),
    ]
    pd.DataFrame(rows).to_csv(namespace / "codexuw_replay_detail.csv", index=False)

    history = load_replay_edge_history(tmp_path, asof="2026-02-01", history_namespace="accepted_credit_history")

    assert history["ticker"].tolist() == ["KEPT"]
