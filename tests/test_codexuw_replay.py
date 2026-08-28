from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.engine import replay_quality_pattern
from codexuw.replay import (
    _guard_result,
    apply_replay_decision_selection,
    build_daily_opportunity_coverage,
    dated_folders,
    simulate_spread_exit,
    write_replay_asof_report,
)


def _row() -> pd.Series:
    return pd.Series(
        {
            "asof": dt.date(2026, 1, 1),
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
            "asof": dt.date(2026, 1, 1),
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


def test_dated_folders_prefers_one_canonical_directory_per_market_day(tmp_path) -> None:
    canonical = tmp_path / "2026-05-19"
    overlay = tmp_path / "2026-05-19-v3-overlay-2026-05-20-live"
    other = tmp_path / "2026-05-20"
    weekend = tmp_path / "2026-05-23"
    juneteenth = tmp_path / "2026-06-19"
    for path in (overlay, canonical, other, weekend, juneteenth):
        path.mkdir()

    folders = dated_folders(tmp_path, None, None)

    assert folders == [canonical, other]


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
    assert result["signal_day"] == dt.date(2026, 1, 1)
    assert result["entry_day"] == dt.date(2026, 1, 2)
    assert result["entry_timing"] == "next_session_hot_chain_eod_fallback"
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


def test_simulate_spread_exit_never_uses_signal_day_quote_for_entry() -> None:
    row = _row()
    quotes = {
        dt.date(2026, 1, 1): {
            "XYZ260116C00105000": _quote(4.00, 4.20),
            "XYZ260116C00110000": _quote(0.10, 0.20),
        },
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
        row,
        close_history={},
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )

    assert result["entry_day"] == dt.date(2026, 1, 2)
    assert round(result["entry_credit"], 4) == 1.08


def test_simulate_spread_exit_does_not_skip_missing_next_session_legs() -> None:
    row = _row()
    close_history = {
        dt.date(2026, 1, 2): pd.DataFrame(
            [{"ticker": "XYZ", "close": 101.0, "sector": "Technology"}]
        ),
        dt.date(2026, 1, 5): pd.DataFrame(
            [{"ticker": "XYZ", "close": 99.0, "sector": "Technology"}]
        ),
    }
    quotes = {
        dt.date(2026, 1, 5): {
            "XYZ260116C00105000": _quote(2.00, 2.20),
            "XYZ260116C00110000": _quote(0.80, 1.00),
        }
    }

    result = simulate_spread_exit(
        row,
        close_history=close_history,
        quote_history=quotes,
        slippage_pct=0.10,
        profit_take_pct=0.60,
        stop_loss_mult=2.0,
    )

    assert result["entry_day"] == dt.date(2026, 1, 2)
    assert result["exact_evaluated"] is False
    assert result["exact_reason"] == "missing_entry_leg_quote"


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


def test_replay_quality_pattern_accepts_validated_credit_and_dte() -> None:
    passed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.25,
        distance_pct=0.032,
        expected_move=0.04,
        dte=30,
        iv_rank=45.0,
    )
    assert passed is True
    assert reason == "validated_credit25_30_dte28_ivrank30"


def test_replay_quality_pattern_matches_live_credit_policy_gates() -> None:
    """The replay guard must never be looser than the live credit policy.

    If these drift apart, the evidence base describes trades the live pipeline
    can never take and every downstream calibration becomes unrepresentative.
    """
    from codexuw.credit_policy import MIN_DTE, MIN_IV_RANK

    failed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.25,
        distance_pct=0.032,
        expected_move=0.04,
        dte=MIN_DTE - 1,
        iv_rank=MIN_IV_RANK + 10,
    )
    assert failed is False
    assert reason == f"replay_guard_dte_below_{int(MIN_DTE)}"

    failed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.25,
        distance_pct=0.032,
        expected_move=0.04,
        dte=MIN_DTE + 1,
        iv_rank=MIN_IV_RANK - 1,
    )
    assert failed is False
    assert reason == f"replay_guard_iv_rank_below_{int(MIN_IV_RANK)}"

    passed, _ = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.25,
        distance_pct=0.032,
        expected_move=0.04,
        dte=MIN_DTE,
        iv_rank=MIN_IV_RANK,
    )
    assert passed is True


def test_replay_quality_pattern_rejects_unvalidated_low_credit_range_trade() -> None:
    passed, reason = replay_quality_pattern(
        direction="Bear Call",
        trend="range",
        credit_pct=0.17,
        distance_pct=0.026,
        expected_move=0.04,
        dte=30,
        iv_rank=45.0,
    )
    assert passed is False
    assert reason == "replay_guard_credit_below_validated_band"


def test_replay_quality_pattern_ignores_distance_buffer() -> None:
    """Short-strike distance is deliberately no longer a gate.

    It is collinear with the credit band (corr -0.734) and ranked worse than no
    selection at all out-of-sample, so a tight-but-in-band spread must pass.
    """
    passed, _ = replay_quality_pattern(
        direction="Bear Call",
        trend="uptrend",
        credit_pct=0.25,
        distance_pct=0.02,
        expected_move=0.04,
        dte=30,
        iv_rank=45.0,
    )
    assert passed is True


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
                "replay_guard_reason": "validated_credit25_30_expected_buffer",
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
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.27,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.30,
                "realized_volatility_30d": 0.20,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.12,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                    "ticker": "BBB",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.20,
                "realized_volatility_30d": 0.30,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.05,
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert selected["decision_pass"].sum() == 1
    assert bool(selected.loc[selected["ticker"].eq("AAA"), "decision_pass"].iloc[0]) is True
    assert selected.loc[selected["ticker"].eq("BBB"), "decision_reason"].iloc[0] == (
        "decision_credit_policy:flow_alignment_below_0.10|iv_hv_ratio_below_0.90"
    )


def test_decision_selection_blocks_near_earnings() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-20",
                "ticker": "META",
                "direction": "Bear Call",
                "stock_price_eod": 670.0,
                "short_strike_eod": 720.0,
                "entry_credit_pct_width": 0.25,
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
                "dte": 22,
                "next_earnings_dt": "2026-08-01",
            }
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert bool(selected["decision_pass"].iloc[0]) is True
    assert selected["decision_reason"].iloc[0] == "decision_selected_independent_debit_sleeve"


def test_daily_opportunity_coverage_distinguishes_ranking_and_guard_misses() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-20",
                "ticker": "SELECTED",
                "strategy": "Bull Call Debit Spread",
                "direction": "Bull Call",
                "exact_evaluated": True,
                "replay_guard_pass": True,
                "decision_pass": True,
                "pnl_1x": 50.0,
            },
            {
                "asof": "2026-04-21",
                "ticker": "RANKMISS",
                "strategy": "Bear Call Credit Spread",
                "direction": "Bear Call",
                "exact_evaluated": True,
                "replay_guard_pass": True,
                "decision_pass": False,
                "decision_reason": "lower_rank",
                "pnl_1x": 75.0,
            },
            {
                "asof": "2026-04-22",
                "ticker": "GUARDMISS",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "exact_evaluated": True,
                "replay_guard_pass": False,
                "replay_guard_reason": "guarded_out",
                "decision_pass": False,
                "pnl_1x": 80.0,
            },
            {
                "asof": "2026-04-23",
                "ticker": "LOSS",
                "strategy": "Bull Call Debit Spread",
                "direction": "Bull Call",
                "exact_evaluated": True,
                "replay_guard_pass": True,
                "decision_pass": True,
                "pnl_1x": -25.0,
            },
        ]
    )

    coverage = build_daily_opportunity_coverage(detail).set_index("asof")

    assert coverage.loc["2026-04-20", "coverage_classification"] == "selected_profitable"
    assert coverage.loc["2026-04-21", "coverage_classification"] == "ranking_miss"
    assert coverage.loc["2026-04-22", "coverage_classification"] == "guard_miss"
    assert coverage.loc["2026-04-23", "coverage_classification"] == "no_profitable_exact_candidate"
    assert coverage.loc["2026-04-22", "best_profitable_ticker"] == "GUARDMISS"


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
        "dte": 22,
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


def test_decision_selection_requires_volatility_richness_even_with_high_iv_rank() -> None:
    """IV/HV is the binding volatility bound; a high iv_rank cannot rescue implied
    vol that is cheaper than the realised vol of the underlying."""
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-22",
                    "ticker": "SEC",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 104.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.60,
                "iv_rank": 55.0,
                "dte": 30,
                "combined_flow_bias": -0.20,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-23",
                    "ticker": "PRI",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.30,
                "realized_volatility_30d": 0.20,
                "iv_rank": 12.0,
                "dte": 30,
                "combined_flow_bias": -0.12,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-23",
                    "ticker": "SEC2",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 104.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.60,
                "iv_rank": 55.0,
                "dte": 30,
                "combined_flow_bias": -0.20,
                "next_earnings_dt": "2026-06-10",
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=1)

    assert bool(selected.loc[selected["ticker"].eq("SEC"), "decision_pass"].iloc[0]) is False
    assert "iv_hv_ratio_below_0.90" in selected.loc[
        selected["ticker"].eq("SEC"), "decision_reason"
    ].iloc[0]
    assert bool(selected.loc[selected["ticker"].eq("PRI"), "decision_pass"].iloc[0]) is True
    assert bool(selected.loc[selected["ticker"].eq("SEC2"), "decision_pass"].iloc[0]) is False
    assert "iv_hv_ratio_below_0.90" in selected.loc[
        selected["ticker"].eq("SEC2"), "decision_reason"
    ].iloc[0]


def test_decision_selection_honors_explicit_one_credit_cap() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                    "ticker": "TOP",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.30,
                "realized_volatility_30d": 0.20,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.12,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                    "ticker": "ADD",
                    "direction": "Bear Call",
                    "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.30,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.20,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                    "ticker": "SKIP",
                    "direction": "Bull Put",
                    "regime": "downtrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 94.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.30,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": 0.20,
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(
        detail,
        max_selected_per_day=8,
        max_credit_selected_per_day=1,
    )

    credit = selected[selected["ticker"].isin(["TOP", "ADD", "SKIP"])]
    assert credit["decision_pass"].sum() == 1
    assert selected.loc[selected["decision_pass"].eq(True), "ticker"].iloc[0] in {"TOP", "ADD", "SKIP"}


def test_decision_selection_takes_all_policy_clear_credits_when_uncapped() -> None:
    detail = pd.DataFrame(
        [
            {
                "asof": "2026-04-29",
                "ticker": "TOP",
                "direction": "Bear Call",
                "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.30,
                "realized_volatility_30d": 0.20,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.12,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                "ticker": "ADD",
                "direction": "Bear Call",
                "regime": "uptrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 110.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.30,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": -0.20,
                "exact_evaluated": True,
            },
            {
                "asof": "2026-04-29",
                "ticker": "SKIP",
                "direction": "Bull Put",
                "regime": "downtrend",
                "stock_price_eod": 100.0,
                "short_strike_eod": 94.0,
                "entry_credit_pct_width": 0.26,
                "entry_quote_width_pct": 0.10,
                "iv30d": 0.45,
                "realized_volatility_30d": 0.30,
                "dte": 30,
                "iv_rank": 45.0,
                "combined_flow_bias": 0.20,
                "exact_evaluated": True,
            },
        ]
    )

    selected = apply_replay_decision_selection(detail, max_selected_per_day=0)

    assert set(selected.loc[selected["decision_pass"].eq(True), "ticker"]) == {"TOP", "ADD", "SKIP"}
