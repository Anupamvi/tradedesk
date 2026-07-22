from codexuw.debit_policy import assess_debit_spread, debit_spread_confidence


def _bull_row(**overrides):
    row = {
        "direction": "Bull Call",
        "bot_flow_source_status": "missing_bot_eod_dp_equity_only",
        "flow_quality": "directional",
        "regime": "uptrend",
        "dte": 28,
        "entry_debit_pct_width": 0.35,
        "reward_risk": 1.6,
        "expected_move_ratio": 1.3,
        "combined_flow_bias": 0.30,
        "entry_quote_width_pct": 0.15,
        "iv_rank": 40.0,
        "oi_carryover_status": "matched_unconfirmed",
        "edge_sample_size": 12,
        "edge_profit_factor": 1.25,
        "edge_avg_pnl": 20.0,
    }
    row.update(overrides)
    return row


def test_directional_contract_flow_can_qualify_without_bot_aggregate():
    ok, reasons = assess_debit_spread(_bull_row(), live=True)
    assert ok, reasons


def test_missing_bot_and_nondirectional_contract_flow_is_rejected():
    ok, reasons = assess_debit_spread(_bull_row(flow_quality="unclear"), live=False)
    assert not ok
    assert "side_aware_bot_or_directional_contract_flow_required" in reasons


def test_spread_leg_contract_is_rejected_even_with_bot_flow():
    ok, reasons = assess_debit_spread(
        _bull_row(bot_flow_source_status="bot_eod_loaded", flow_quality="spread_leg"),
        live=False,
    )
    assert not ok
    assert "contract_flow_spread_leg" in reasons


def test_bull_call_intermediate_maturity_gap_is_rejected():
    ok, reasons = assess_debit_spread(_bull_row(dte=15), live=False)
    assert not ok
    assert "dte_outside_7_10_or_22_45" in reasons


def test_bear_put_requires_validated_directional_flow_floor():
    ok, reasons = assess_debit_spread(
        _bull_row(
            direction="Bear Put",
            regime="downtrend",
            dte=28,
            combined_flow_bias=-0.19,
        ),
        live=False,
    )
    assert not ok
    assert "flow_alignment_below_0.20" in reasons

    ok, reasons = assess_debit_spread(
        _bull_row(
            direction="Bear Put",
            regime="downtrend",
            dte=28,
            combined_flow_bias=-0.20,
        ),
        live=False,
    )
    assert ok, reasons


def test_complete_split_bot_bundle_counts_as_full_flow():
    high, reasons = debit_spread_confidence(
        _bull_row(
            bot_flow_source_status="bot_eod_split_bundle_loaded",
            flow_quality="directional",
            oi_carryover_status="supportive",
            edge_sample_size=24,
            edge_profit_factor=1.50,
            edge_match_level="strategy_regime",
        ),
        live=True,
    )
    assert high == "high"
    assert reasons == []


def test_medium_and_high_live_tiers_are_explicit():
    medium, medium_reasons = debit_spread_confidence(_bull_row(), live=True)
    assert medium == "medium"
    assert "high_requires_full_bot_flow" in medium_reasons

    high, high_reasons = debit_spread_confidence(
        _bull_row(
            bot_flow_source_status="bot_eod_loaded",
            flow_quality="directional",
            oi_carryover_status="supportive",
            edge_sample_size=24,
            edge_profit_factor=1.50,
            edge_match_level="strategy_regime",
        ),
        live=True,
    )
    assert high == "high"
    assert high_reasons == []
