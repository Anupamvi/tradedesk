from __future__ import annotations

import pandas as pd

from codexuw.range_gex_income_book import (
    build_vertical_shadow,
    derive_gex_features,
    enrich_replay,
    evaluate_shadow_book,
)


def _gex() -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "ticker": "XYZ",
                "date": "2026-01-05",
                "captured_utc": "2026-01-06T01:00:00Z",
                "spot": 100.0,
                "gamma_oi_per_1pct": 1_000_000.0,
                "gamma_vol_per_1pct": 200_000.0,
                "gamma_dir_per_1pct": 50_000.0,
            }
        ]
    )
    strikes = pd.DataFrame(
        [
            {"ticker": "XYZ", "date": "2026-01-05", "spot": 100.0, "strike": 90.0, "call_gamma_oi": 5.0, "put_gamma_oi": -10.0},
            {"ticker": "XYZ", "date": "2026-01-05", "spot": 100.0, "strike": 95.0, "call_gamma_oi": 10.0, "put_gamma_oi": -80.0},
            {"ticker": "XYZ", "date": "2026-01-05", "spot": 100.0, "strike": 105.0, "call_gamma_oi": 100.0, "put_gamma_oi": -10.0},
            {"ticker": "XYZ", "date": "2026-01-05", "spot": 100.0, "strike": 110.0, "call_gamma_oi": 10.0, "put_gamma_oi": -5.0},
        ]
    )
    return derive_gex_features(summary, strikes)


def test_derive_gex_features_tracks_walls_and_capture_timing() -> None:
    result = _gex().iloc[0]

    assert result["gex_call_wall"] == 105.0
    assert result["gex_put_wall"] == 95.0
    assert bool(result["gex_spot_between_walls"])
    assert result["gex_capture_timing"] == "point_in_time"
    assert result["gex_capture_lag_days"] == 1


def test_vertical_shadow_requires_range_positive_gamma_and_outside_wall() -> None:
    rows = []
    for strategy, short, flow, credit in [
        ("Bear Call Credit Spread", 110.0, -0.02, 1.50),
        ("Bull Put Credit Spread", 90.0, 0.02, 1.00),
    ]:
        rows.append(
            {
                "asof": "2026-01-05",
                "entry_day": "2026-01-06",
                "exit_day": "2026-01-20",
                "ticker": "XYZ",
                "strategy": strategy,
                "regime": "range",
                "expiry": "2026-02-06",
                "exact_evaluated": True,
                "earnings_crosses": False,
                "entry_credit": credit,
                "entry_width": 5.0,
                "entry_credit_pct_width": credit / 5.0,
                "entry_quote_width_pct": 0.10,
                "expected_move_ratio": 0.50,
                "entry_dte": 31,
                "combined_flow_bias": flow,
                "pnl_1x": 50.0,
                "short_strike_eod": short,
                "long_strike_eod": short + (5.0 if "Call" in strategy else -5.0),
                "iv_hv_ratio": 1.20,
            }
        )
    enriched = enrich_replay(pd.DataFrame(rows), _gex())
    qualified, selected = build_vertical_shadow(enriched)

    assert len(qualified) == 2
    assert len(selected) == 1
    assert selected.iloc[0]["strategy"] == "Bear Call Credit Spread"


def test_shadow_book_cannot_pass_without_real_holdout_or_point_in_time_gex() -> None:
    selected = pd.DataFrame(
        [
            {
                "entry_day": "2026-01-06",
                "exit_day": "2026-01-20",
                "entry_credit": 1.0,
                "pnl_1x": 50.0,
                "gex_capture_timing": "historical_api_reconstruction",
            }
        ]
    )
    summary, metrics, _ = evaluate_shadow_book(
        selected,
        cutoff=pd.Timestamp("2026-05-19"),
        credit_column="entry_credit",
    )

    assert summary["status"] == "RESEARCH_ONLY"
    assert not summary["execution_authorized"]
    assert "historical GEX was reconstructed after the source date" in summary["reasons"]
    assert "untouched holdout sample below threshold" in summary["reasons"]
    assert len(metrics) == 9
