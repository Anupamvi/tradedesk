import pandas as pd

from codexuw.credit_policy import (
    MAX_CREDIT_PCT_WIDTH,
    MAX_DTE,
    MIN_CREDIT_PCT_WIDTH,
    MIN_DTE,
)
from scripts.strategy_universe import (
    HISTORICAL_STRATEGY_SPECS,
    MAX_DEBIT_PCT_WIDTH,
    build_sector_state,
    build_structure,
    liquidate_structure,
)
from scripts.sector_strategy_grid import build_matched_grid


def _quotes() -> pd.DataFrame:
    rows = []
    # 35 DTE serves the verticals' live 22-45 band; 80/108 serve the generic
    # 60-110 band and its far leg. Prices are set so every vertical lands inside
    # the live credit/debit-to-width band.
    for expiry, dte in (("2026-09-04", 35), ("2026-12-18", 80), ("2027-01-15", 108)):
        for option_type, strikes in (("call", [100, 105, 110, 115]), ("put", [85, 90, 95, 100])):
            for strike in strikes:
                bid = max(0.2, 6.0 - abs(strike - 100) * 0.30)
                rows.append(
                    {
                        "ticker": "AAA",
                        "expiry": expiry,
                        "dte": dte,
                        "stock_price": 100.0,
                        "option_type": option_type,
                        "option_symbol": f"AAA-{expiry}-{option_type}-{strike}",
                        "strike": float(strike),
                        "last_bid": bid,
                        "last_ask": bid + 0.3,
                        "curr_oi": 1_000,
                        "spread_pct": 0.05,
                    }
                )
    return pd.DataFrame(rows)


def test_vertical_specs_mirror_live_execution_policy() -> None:
    """A study run under looser rules than live describes untradable structures."""
    by_key = {spec.key: spec for spec in HISTORICAL_STRATEGY_SPECS}
    for key in ("bull_put_credit_vertical", "bear_call_credit_vertical"):
        spec = by_key[key]
        assert spec.dte_band == (MIN_DTE, MAX_DTE), key
        assert spec.entry_pct_width_band == (MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH), key
        assert spec.screen_earnings_before_expiry, key
        assert spec.hold_days < MIN_DTE, key
    for key in ("bull_call_debit_vertical", "bear_put_debit_vertical"):
        spec = by_key[key]
        assert spec.entry_pct_width_band == (0.0, MAX_DEBIT_PCT_WIDTH), key
        assert spec.screen_earnings_before_expiry, key
        assert spec.hold_days < spec.dte_band[0], key


def test_entry_band_and_earnings_screen_reject_untradable_structures() -> None:
    spec = next(s for s in HISTORICAL_STRATEGY_SPECS if s.key == "bear_call_credit_vertical")
    quotes = _quotes()
    assert len(build_structure(quotes, {"AAA"}, spec)) == 1

    # Credit below the live 25%-of-width floor must not enter the study.
    thin = quotes.copy()
    short_leg = thin.option_type.eq("call") & thin.strike.eq(105.0)
    thin.loc[short_leg, "last_bid"] = thin.loc[short_leg, "last_bid"] - 2.5
    assert build_structure(thin, {"AAA"}, spec).empty

    # Earnings before expiry is a live hard blocker.
    earnings = pd.Series({"AAA": pd.Timestamp("2026-08-20")})
    assert build_structure(quotes, {"AAA"}, spec, earnings_by_ticker=earnings).empty
    later = pd.Series({"AAA": pd.Timestamp("2026-11-01")})
    assert len(build_structure(quotes, {"AAA"}, spec, earnings_by_ticker=later)) == 1


def test_all_historical_structures_use_exact_expiry_models_and_cashflows() -> None:
    quotes = _quotes()
    assert len(HISTORICAL_STRATEGY_SPECS) == 32
    for spec in HISTORICAL_STRATEGY_SPECS:
        built = build_structure(quotes, {"AAA"}, spec)
        assert len(built) == 1, spec.key
        assert built.iloc[0]["max_risk_per_share"] > 0
        if any(leg.expiry_slot == "far" for leg in spec.legs):
            assert pd.notna(built.iloc[0]["far_expiry"]), spec.key
            assert built.iloc[0]["far_expiry"] != built.iloc[0]["expiry"], spec.key
        else:
            assert pd.isna(built.iloc[0]["far_expiry"]), spec.key
        marked = liquidate_structure(built, quotes, spec)
        assert len(marked) == 1, spec.key
        assert marked.iloc[0]["pnl"] < 0


def test_sector_state_is_point_in_time_and_dynamic() -> None:
    rows = []
    for index, date in enumerate(pd.date_range("2026-01-02", periods=6)):
        for ticker, offset in (("A1", 0.00), ("A2", -0.05)):
            rows.append({"date": str(date.date()), "sector": "A", "ticker": ticker, "pos_52w": 0.40 + index * 0.09 + offset, "ret_1d": 0.03, "flow_escalation": 2.0})
        for ticker, offset in (("B1", 0.00), ("B2", 0.05)):
            rows.append({"date": str(date.date()), "sector": "B", "ticker": ticker, "pos_52w": 0.70 - index * 0.09 + offset, "ret_1d": -0.02, "flow_escalation": 0.5})
    panel = pd.DataFrame(rows)

    state = build_sector_state(panel)
    state = state[state["date"].eq("2026-01-07")].set_index("sector")

    assert state.loc["A", "sector_state"] == "emerging"
    assert state.loc["A", "sector_momentum_change_5s"] > 0
    assert state.loc["A", "sector_emergence_score"] > state.loc["B", "sector_emergence_score"]


def test_matched_grid_draws_controls_after_construction() -> None:
    universe = pd.DataFrame(
        [
            {"signal_date": "2026-01-02", "sector": "A", "strategy": "long_call", "ticker": "A1", "signal_selected": True},
            {"signal_date": "2026-01-02", "sector": "A", "strategy": "long_call", "ticker": "A2", "signal_selected": False},
            {"signal_date": "2026-01-02", "sector": "A", "strategy": "long_call", "ticker": "A3", "signal_selected": False},
        ]
    )

    grid = build_matched_grid(universe)

    assert grid["mode"].value_counts().to_dict() == {"signal": 1, "random": 1}