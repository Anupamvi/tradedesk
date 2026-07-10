import datetime as dt

import pandas as pd

from codexuw.replay import apply_replay_decision_selection


def test_debit_sleeve_does_not_displace_same_day_credit_selection():
    asof = dt.date(2026, 4, 20)
    rows = [
        {
            "asof": asof,
            "ticker": "CREDIT",
            "direction": "Bull Put",
            "strategy": "Bull Put Credit Spread",
            "exact_fillable": True,
            "exact_evaluated": True,
            "entry_credit_pct_width": 0.20,
            "entry_quote_width_pct": 0.10,
            "stock_price_eod": 100.0,
            "short_strike_eod": 90.0,
            "iv30d": 0.20,
            "combined_flow_bias": 0.20,
            "dte": 30,
        },
        {
            "asof": asof,
            "ticker": "DEBIT",
            "direction": "Bull Call",
            "strategy": "Bull Call Debit Spread",
            "exact_fillable": True,
            "exact_evaluated": True,
            "entry_debit_pct_width": 0.35,
            "breakeven_distance_pct": 0.04,
            "reward_risk": 1.80,
            "iv30d": 0.30,
            "iv_rank": 40.0,
            "combined_flow_bias": 0.30,
            "bot_flow_source_status": "bot_eod_loaded",
            "flow_quality": "directional",
            "regime": "uptrend",
            "entry_quote_width_pct": 0.10,
            "dte": 28,
        },
    ]

    selected = apply_replay_decision_selection(
        pd.DataFrame(rows),
        max_selected_per_day=1,
        max_debit_selected_per_day=1,
    )

    assert selected.set_index("ticker")["decision_pass"].to_dict() == {"CREDIT": True, "DEBIT": True}
    assert selected.set_index("ticker").at["DEBIT", "decision_reason"] == "decision_selected_independent_debit_sleeve"
