from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.engine import replay_quality_pattern
from codexuw.replay import _guard_result, apply_replay_decision_selection, simulate_spread_exit, write_replay_asof_report


def _row() -> pd.Series:
    return pd.Series(
        {
            "asof": dt.date(2026, 1, 2),
            "expiry": dt.date(2026, 1, 16),
            "ticker": "XYZ",
            "direction": "Bear Call",
            "short_strike_eod": 105.0,
            "long_strike_eod": 110.0,
            "short_leg_eod": "XYZ260116C00105000",
            "long_leg_eod": "XYZ260116C00110000",
        }
    )


def _debit_row() -> pd.Series:
    return pd.Series(
        {
            "asof": dt.date(2026, 1, 2),
            "expiry": dt.date(2026, 1, 16),
            "ticker": "XYZ",
            "direction": "Bull Call",
            "strategy": "Bull Call Debit Spread",
            "stock_price_eod": 100.0,
            "long_strike_eod": 100.0,
            "short_strike_eod": 105.0,
            "long_leg_eod": "XYZ260116C00100000",
            "short_leg_eod": "XYZ260116C00105000",
            "combined_flow_bias": 0.12,
            "iv30d": 0.30,
            "dte": 14,
        }
    )


def _quote(bid: float, ask: float) -> dict[str, float | str]:
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "volume": 1000.0, "open_interest": 1000.0}


def test_simulate_spread_exit_hits_profit_target() -> None:
    quotes = {
        dt.date(2026, 1, 2): {
            "XYZ260116C00105000": _quote(2.00, 2.20),
            "XYZ260116C00110000": _quote(0.80, 1.00),
        },
        dt.date(2026, 1, 5): {
            "XYZ260116C00105000": _quote(1.00, 1.10),
            "XYZ260116C00110000": _quote(0.85, 0.95),
        },
    }
    result = simulate_spread_exit(
        _row(),
        close_history={},
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )
    assert result["exact_evaluated"] is True
    assert result["exit_reason"] == "profit_target"
    assert result["pnl_1x"] > 0


def test_simulate_spread_exit_hits_stop_loss() -> None:
    quotes = {
        dt.date(2026, 1, 2): {
            "XYZ260116C00105000": _quote(2.00, 2.20),
            "XYZ260116C00110000": _quote(0.80, 1.00),
        },
        dt.date(2026, 1, 5): {
            "XYZ260116C00105000": _quote(3.00, 3.20),
            "XYZ260116C00110000": _quote(0.50, 0.70),
        },
    }
    result = simulate_spread_exit(
        _row(),
        close_history={},
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )
    assert result["exact_evaluated"] is True
    assert result["exit_reason"] == "stop_loss"
    assert result["pnl_1x"] < 0


def test_simulate_debit_spread_exit_hits_profit_target() -> None:
    quotes = {
        dt.date(2026, 1, 2): {
            "XYZ260116C00100000": _quote(2.00, 2.20),
            "XYZ260116C00105000": _quote(0.80, 1.00),
        },
        dt.date(2026, 1, 5): {
            "XYZ260116C00100000": _quote(4.00, 4.20),
            "XYZ260116C00105000": _quote(1.50, 1.70),
        },
    }

    result = simulate_spread_exit(
        _debit_row(),
        close_history={},
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )

    assert result["exact_evaluated"] is True
    assert result["entry_side"] == "debit"
    assert result["exit_reason"] == "profit_target"
    assert result["exit_value"] > result["entry_debit"]
    assert result["pnl_1x"] > 0


def test_debit_above_target_is_annotated_not_unfilled() -> None:
    quotes = {
        dt.date(2026, 1, 2): {
            "XYZ260116C00100000": _quote(3.10, 3.30),
            "XYZ260116C00105000": _quote(0.60, 0.80),
        },
        dt.date(2026, 1, 5): {
            "XYZ260116C00100000": _quote(6.00, 6.20),
            "XYZ260116C00105000": _quote(0.80, 1.00),
        },
    }

    result = simulate_spread_exit(
        _debit_row(),
        close_history={},
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )

    assert result["exact_evaluated"] is True
    assert result["entry_price_annotation"] == "entry_debit_above_target"
    assert "above_target" in result["fill_reason"]
    assert result["entry_debit"] > result["target_debit"]


def test_replay_quality_pattern_accepts_validated_credit_and_buffer() -> None:
    passed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.20,
        distance_pct=0.027,
        expected_move=0.04,
    )
    assert passed is True
    assert reason == "validated_credit18_30_expected_buffer"


def test_replay_quality_pattern_rejects_unvalidated_low_credit_range_trade() -> None:
    passed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="range",
        credit_pct=0.17,
        distance_pct=0.026,
        expected_move=0.04,
    )
    assert passed is False
    assert reason == "replay_guard_credit_below_validated_band"


def test_replay_quality_pattern_rejects_credit_without_expected_move_buffer() -> None:
    passed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.24,
        distance_pct=0.02,
        expected_move=0.04,
    )
    assert passed is False
    assert reason == "replay_guard_insufficient_expected_move_buffer"


def test_write_replay_asof_report_emits_explicit_trade_ticket(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                "ticker": "MSFT",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "expiry": "2026-05-15",
                "dte": 16,
                "stock_price_eod": 430.0,
                "short_strike_eod": 450.0,
                "long_strike_eod": 455.0,
                "short_leg_eod": "MSFT260515C00450000",
                "long_leg_eod": "MSFT260515C00455000",
                "entry_width": 5.0,
                "entry_credit": 1.08,
                "entry_credit_pct_width": 0.216,
                "entry_quote_width_pct": 0.12,
                "iv30d": 0.25,
                "flow_total_premium": 100_000_000.0,
                "combined_flow_bias": -0.08,
                "edge_type": "flow+volatility",
                "exact_evaluated": True,
                "replay_guard_pass": True,
                "replay_guard_reason": "validated_credit18_30_expected_buffer",
                "exit_reason": "profit_target",
                "exit_day": "2026-04-30",
                "pnl_1x": 90.95,
            }
        ]
    )

    report = write_replay_asof_report(detail, tmp_path, dt.date(2026, 4, 29))
    text = report.read_text(encoding="utf-8")

    assert "Sell Leg" in text
    assert "Buy Leg" in text
    assert "MSFT 2026-05-15 450C" in text
    assert "MSFT260515C00450000" not in text
    assert "🟢" in text
    assert "Trade Conviction" in text
    assert "profit_target on 2026-04-30" in text


def test_decision_selection_keeps_only_strongest_flow_aligned_trade_per_day() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                "ticker": "AAA",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.20,
                "dte": 21,
                "combined_flow_bias": -0.12,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                "ticker": "BBB",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.20,
                "dte": 21,
                "combined_flow_bias": -0.05,
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert selected["decision_pass"].sum() == 1
    assert bool(selected.loc[selected["ticker"].eq("AAA"), "decision_pass"].iloc[0]) is True
    assert selected.loc[selected["ticker"].eq("BBB"), "decision_reason"].iloc[0] == "decision_credit_policy:flow_alignment_below_0.10"


def test_decision_selection_blocks_near_earnings() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-20",
                "ticker": "META",
                "direction": "Bear Call",
                "stock_price_eod": 670.0,
                "short_strike_eod": 720.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.40,
                "dte": 11,
                "combined_flow_bias": -0.12,
                "next_earnings_dt": "2026-04-29",
                "exact_evaluated": True,
            }
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert selected["decision_pass"].sum() == 0
    assert selected["decision_reason"].iloc[0] == "decision_earnings_within_10d:9"


def test_decision_selection_retains_above_target_debit_as_annotation() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                "ticker": "DEB",
                "direction": "Bull Call",
                "strategy": "Bull Call Debit Spread",
                "stock_price_eod": 100.0,
                "long_strike_eod": 100.0,
                "short_strike_eod": 105.0,
                "entry_debit": 2.60,
                "entry_debit_pct_width": 0.52,
                "entry_quote_width_pct": 0.10,
                "reward_risk": 0.92,
                "breakeven_distance_pct": 0.026,
                "iv30d": 0.45,
                "dte": 21,
                "combined_flow_bias": 0.14,
                "exact_evaluated": True,
                "exact_win": True,
                "pnl_1x": 110.0,
            }
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert selected["decision_pass"].sum() == 0
    assert selected["decision_reason"].iloc[0] == "decision_debit_above_target_watch_annotation"
    assert selected["decision_tier"].iloc[0] == "debit_watch_annotation"


def test_decision_selection_never_reads_future_debit_outcome() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": dt.date(2026, 4, 20),
                "ticker": "LOSS",
                "direction": "Bull Call",
                "strategy": "Bull Call Debit Spread",
                "exact_fillable": True,
                "exact_evaluated": True,
                "exact_win": False,
                "pnl_1x": -125.0,
                "entry_debit_pct_width": 0.25,
                "breakeven_distance_pct": 0.05,
                "reward_risk": 1.6,
                "iv30d": 0.30,
                    "combined_flow_bias": 0.20,
                    "bot_flow_source_status": "bot_eod_loaded",
                    "flow_quality": "directional",
                    "regime": "uptrend",
                    "entry_quote_width_pct": 0.10,
                "dte": 21,
                "next_earnings_dt": "2026-08-01",
            }
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert bool(selected["decision_pass"].iloc[0]) is True
    assert selected["decision_reason"].iloc[0] == "decision_selected_independent_debit_sleeve"


def test_replay_guard_never_uses_future_debit_pnl() -> None:
    record = {
        "direction": "Bull Call",
        "strategy": "Bull Call Debit Spread",
        "exact_fillable": True,
        "exact_evaluated": True,
        "pnl_1x": -150.0,
        "entry_debit": 1.50,
        "entry_width": 5.0,
        "entry_debit_pct_width": 0.30,
        "breakeven_distance_pct": 0.05,
        "reward_risk": 2.0,
        "entry_quote_width_pct": 0.10,
        "iv30d": 0.30,
        "combined_flow_bias": 0.20,
        "bot_flow_source_status": "bot_eod_loaded",
        "regime": "uptrend",
        "dte": 21,
        "iv_rank": 40.0,
        "flow_quality": "directional",
    }

    passed, reason = _guard_result(record)

    assert passed is True
    assert reason == "validated_debit_replay_edge"


def test_debit_time_stop_closes_before_expiration() -> None:
    row = _row().copy()
    row["direction"] = "Bull Call"
    row["strategy"] = "Bull Call Debit Spread"
    row["long_strike_eod"] = 100.0
    row["short_strike_eod"] = 105.0
    row["long_leg_eod"] = "XYZ260116C00100000"
    row["short_leg_eod"] = "XYZ260116C00105000"
    quote_history = {
        dt.date(2026, 1, 2): {
            "XYZ260116C00100000": {"bid": 2.90, "ask": 3.10, "mark": 3.00, "mid": 3.00},
            "XYZ260116C00105000": {"bid": 1.40, "ask": 1.60, "mark": 1.50, "mid": 1.50},
        },
        dt.date(2026, 1, 9): {
            "XYZ260116C00100000": {"bid": 2.50, "ask": 2.70, "mark": 2.60, "mid": 2.60},
            "XYZ260116C00105000": {"bid": 1.20, "ask": 1.40, "mark": 1.30, "mid": 1.30},
        },
    }

    result = simulate_spread_exit(
        row,
        close_history={},
        quote_history=quote_history,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
        debit_time_stop_dte=7,
    )

    assert result["exact_evaluated"] is True
    assert result["exit_reason"] == "time_stop_7dte"


def test_decision_selection_requires_distance_even_with_volatility_edge() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-22",
                "ticker": "SEC",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 104.0,
                "entry_credit_pct_width": 0.24,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.50,
                "dte": 23,
                "combined_flow_bias": -0.20,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-23",
                "ticker": "PRI",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.20,
                "dte": 21,
                "combined_flow_bias": -0.12,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-23",
                "ticker": "SEC2",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 104.0,
                "entry_credit_pct_width": 0.24,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.50,
                "dte": 23,
                "combined_flow_bias": -0.20,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert bool(selected.loc[selected["ticker"].eq("SEC"), "decision_pass"].iloc[0]) is False
    assert "credit_short_strike_inside_distance_buffer" in selected.loc[
        selected["ticker"].eq("SEC"), "decision_reason"
    ].iloc[0]
    assert bool(selected.loc[selected["ticker"].eq("PRI"), "decision_pass"].iloc[0]) is True
    assert bool(selected.loc[selected["ticker"].eq("SEC2"), "decision_pass"].iloc[0]) is False
    assert "credit_short_strike_inside_distance_buffer" in selected.loc[
        selected["ticker"].eq("SEC2"), "decision_reason"
    ].iloc[0]


def test_decision_selection_caps_credit_edge_sleeve_at_one_per_day() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                "ticker": "TOP",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.20,
                "dte": 21,
                "combined_flow_bias": -0.12,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                "ticker": "ADD",
                "direction": "Bear Call",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.22,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "dte": 21,
                "combined_flow_bias": -0.20,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                "ticker": "SKIP",
                "direction": "Bull Put",
                "stock_price_eod": 100.0,
                "short_strike_eod": 94.0,
                "entry_credit_pct_width": 0.22,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "dte": 21,
                "combined_flow_bias": 0.20,
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=8)

    assert selected.loc[selected["ticker"].eq("TOP"), "decision_pass"].iloc[0]
    assert not selected.loc[selected["ticker"].eq("ADD"), "decision_pass"].iloc[0]
    assert selected.loc[selected["ticker"].eq("ADD"), "decision_reason"].iloc[0] == "decision_eligible"
    assert not selected.loc[selected["ticker"].eq("SKIP"), "decision_pass"].iloc[0]
