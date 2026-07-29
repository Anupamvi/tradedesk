"""Regression tests for the non-executable EV shadow lane."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwos.pattern_analysis_v2 import shadow_lane


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "direction": ["bullish", "bearish"],
            "DTE": [30, 45],
            "bid": [1.0, 2.0],
            "ask": [1.1, 2.2],
            "spread": [0.09, 0.09],
            "slippage": [5.0, 6.0],
            "fees_commissions": [1.3, 1.3],
            "contract_profile": [
                "LONG_OPTION__DTE_14_30__NEAR_OTM",
                "CREDIT_SPREAD__DTE_31_45__FAR_OTM",
            ],
            "pattern_family": [
                "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY",
                "OI_GAMMA_CONTINUATION__BEARISH__CREDIT_SPREAD__ENERGY",
            ],
            "regime_alignment": ["conflicts with RISK_OFF", "aligned/acceptable in RISK_OFF"],
            "strikes": ["100", "90/80"],
            "expiration": ["2026-08-21", "2026-09-18"],
            "entry": ["debit 1.00-1.10", "credit 2.00"],
        }
    )


def _history(rows: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": ["SCORED"] * rows,
            "signal_date": ["2026-07-01"] * rows,
            "ticker": ["AAA", "BBB"] * (rows // 2),
            "direction": ["bullish", "bearish"] * (rows // 2),
            "net_r": [0.5, -0.4] * (rows // 2),
            "dte": [30, 45] * (rows // 2),
            "bid_ask_spread_pct": [0.09, 0.08] * (rows // 2),
            "entry_ask": [1.1, 2.2] * (rows // 2),
            "entry_bid": [1.0, 2.0] * (rows // 2),
            "round_trip_fees": [1.3, 1.3] * (rows // 2),
            "strategy_kind": ["long_option", "credit_spread"] * (rows // 2),
            "sector": ["TECHNOLOGY", "ENERGY"] * (rows // 2),
            "market_regime": ["RISK_OFF", "RISK_OFF"] * (rows // 2),
            "contract_profile": [
                "LONG_OPTION__DTE_14_30__NEAR_OTM",
                "CREDIT_SPREAD__DTE_31_45__FAR_OTM",
            ]
            * (rows // 2),
        }
    )


def test_board_and_history_yield_identical_feature_schema() -> None:
    from_board = shadow_lane.normalize_features(_board(), source="board")
    from_history = shadow_lane.normalize_features(_history(4), source="history")
    assert list(from_board.columns) == list(from_history.columns)
    expected = set(shadow_lane.NUMERIC_FEATURES) | set(shadow_lane.CATEGORICAL_FEATURES)
    assert set(from_board.columns) == expected


def test_board_features_are_derived_from_contract_profile() -> None:
    features = shadow_lane.normalize_features(_board(), source="board")
    assert features["strategy_kind"].tolist() == ["LONG_OPTION", "CREDIT_SPREAD"]
    assert features["dte_bucket"].tolist() == ["DTE_14_30", "DTE_31_45"]
    assert features["moneyness_bucket"].tolist() == ["NEAR_OTM", "FAR_OTM"]
    assert features["sector"].tolist() == ["TECHNOLOGY", "ENERGY"]
    assert features["market_regime"].tolist() == ["RISK_OFF", "RISK_OFF"]


def test_training_never_uses_rows_on_or_after_as_of() -> None:
    history = _history()
    history.loc[history.index[:200], "signal_date"] = "2026-07-20"
    model, rows, through = shadow_lane.train_ev_model(history, "2026-07-20")
    assert model is not None
    assert rows == 200
    assert through < "2026-07-20"


def test_insufficient_history_returns_no_model() -> None:
    model, rows, _ = shadow_lane.train_ev_model(_history(10), "2026-07-20")
    assert model is None
    assert rows == 0


def test_regime_gate_stands_down_conflicting_rows() -> None:
    assert shadow_lane.regime_stand_down(_board()).tolist() == [True, False]


def test_ledger_rows_are_non_executable_and_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    picks = _board().assign(predicted_ev_r=[0.2, 0.1])
    kwargs = dict(
        variant="ungated",
        signal_date="2026-07-23",
        run_date="2026-07-23",
        train_rows=400,
        train_through="2026-07-16",
    )
    assert shadow_lane.append_shadow_rows(ledger, picks, **kwargs) == 2
    assert shadow_lane.append_shadow_rows(ledger, picks, **kwargs) == 0

    rows = pd.read_csv(ledger)
    assert len(rows) == 2
    assert (rows["execution_eligible"].astype(str).str.lower() == "false").all()
    assert (rows["no_order_placement"].astype(str).str.lower() == "true").all()
    assert (rows["status"] == "PENDING").all()

    raw = ledger.read_text(encoding="utf-8")
    assert "execution_eligible" in raw
    assert raw.count(",false,true,") == 2


def test_first_observation_is_immutable(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    kwargs = dict(variant="ungated", signal_date="2026-07-23", train_rows=400, train_through="2026-07-16")
    shadow_lane.append_shadow_rows(
        ledger, _board().assign(predicted_ev_r=[0.2, 0.1]), run_date="2026-07-23", **kwargs
    )
    shadow_lane.append_shadow_rows(
        ledger, _board().assign(predicted_ev_r=[9.9, 8.8]), run_date="2026-07-27", **kwargs
    )
    rows = pd.read_csv(ledger)
    assert len(rows) == 2
    assert rows["predicted_ev_r"].tolist() == [0.2, 0.1]
    assert (rows["first_observed_run_date"] == "2026-07-23").all()


def test_resolution_fills_realized_outcome(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    shadow_lane.append_shadow_rows(
        ledger,
        _board().assign(predicted_ev_r=[0.2, 0.1]),
        variant="ungated",
        signal_date="2026-07-01",
        run_date="2026-07-01",
        train_rows=400,
        train_through="2026-06-30",
    )
    assert shadow_lane.resolve_pending(ledger, _history(4), "2026-07-08") == 2
    rows = pd.read_csv(ledger)
    assert (rows["status"] == "RESOLVED").all()
    assert rows.loc[rows["ticker"] == "AAA", "realized_net_r"].iloc[0] == pytest.approx(0.5)
    assert rows.loc[rows["ticker"] == "BBB", "realized_net_r"].iloc[0] == pytest.approx(-0.4)


def test_summary_never_auto_promotes(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    shadow_lane.append_shadow_rows(
        ledger,
        _board().assign(predicted_ev_r=[0.2, 0.1]),
        variant="ungated",
        signal_date="2026-07-01",
        run_date="2026-07-01",
        train_rows=400,
        train_through="2026-06-30",
    )
    shadow_lane.resolve_pending(ledger, _history(4), "2026-07-08")
    summary = shadow_lane.summarize_ledger(ledger)
    assert summary["ungated"]["resolved"] == 2
    assert summary["ungated"]["promotion_ready"] is False
