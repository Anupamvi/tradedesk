import math
import datetime as dt

import pytest

from uwos.run_mode_a_two_stage import (
    SCHWAB_LIVE_GEX_SOURCE,
    SCHWAB_STALE_GEX_SOURCE,
    compute_schwab_chain_gex,
    entry_gate_strict_pass,
    fire_long_delta_proxy_ok,
    live_gex_entry_block_reason,
    live_mode_date_violation,
    normalize_probability,
    partial_ev_ml_debit,
    pilot_convexity_blockers_allow,
    split_approval_blockers,
    stage2_mode_name,
)


def test_compute_schwab_chain_gex_uses_gamma_oi_multiplier_and_spot():
    payload = {
        "underlying": {
            "mark": 100.0,
            "quoteTime": 1776988798120,
        },
        "callExpDateMap": {
            "2026-05-15:20": {
                "105.0": [{"gamma": 0.02, "openInterest": 50, "multiplier": 100}],
                "110.0": [{"gamma": 0.01, "openInterest": 20, "multiplier": 100}],
            }
        },
        "putExpDateMap": {
            "2026-05-15:20": {
                "95.0": [{"gamma": 0.03, "openInterest": 100, "multiplier": 100}],
                "90.0": [{"gamma": 0.01, "openInterest": 50, "multiplier": 100}],
            }
        },
    }

    result = compute_schwab_chain_gex(payload)

    assert result is not None
    assert result["net_gex"] == -23000.0
    assert result["gex_regime"] == "volatile"
    assert result["gex_support"] == 95.0
    assert result["gex_resistance"] == 105.0
    assert result["gex_source"] == SCHWAB_LIVE_GEX_SOURCE
    assert result["gex_time"]


def test_compute_schwab_chain_gex_requires_spot_and_valid_contracts():
    assert compute_schwab_chain_gex({"callExpDateMap": {}, "putExpDateMap": {}}) is None

    result = compute_schwab_chain_gex(
        {
            "underlyingPrice": 50.0,
            "callExpDateMap": {"2026-05-15:20": {"55.0": [{"gamma": None, "openInterest": 100}]}},
            "putExpDateMap": {},
        }
    )
    assert result is None


def test_compute_schwab_chain_gex_uses_chain_underlying_price_when_mark_missing():
    result = compute_schwab_chain_gex(
        {
            "underlying": {"mark": None, "last": None, "close": None},
            "underlyingPrice": 25.0,
            "callExpDateMap": {"2026-05-15:20": {"30.0": [{"gamma": 0.1, "openInterest": 4}]}},
            "putExpDateMap": {},
        }
    )

    assert result is not None
    assert result["net_gex"] == 1000.0
    assert result["gex_regime"] == "pinned"
    assert math.isnan(result["gex_support"])
    assert result["gex_resistance"] == 30.0


def test_compute_schwab_chain_gex_preserves_source_label_for_stale_snapshots():
    result = compute_schwab_chain_gex(
        {
            "underlyingPrice": 10.0,
            "callExpDateMap": {"2026-05-15:20": {"12.0": [{"gamma": 0.1, "openInterest": 1}]}},
            "putExpDateMap": {},
        },
        source=SCHWAB_STALE_GEX_SOURCE,
    )

    assert result is not None
    assert result["gex_source"] == SCHWAB_STALE_GEX_SOURCE


def test_live_gex_entry_block_requires_current_schwab_chain_gex():
    assert live_gex_entry_block_reason({"gex_source": SCHWAB_LIVE_GEX_SOURCE}, True) == ""
    assert live_gex_entry_block_reason({"gex_source": SCHWAB_STALE_GEX_SOURCE}, False) == ""

    reason = live_gex_entry_block_reason({"gex_source": SCHWAB_STALE_GEX_SOURCE}, True)
    assert "Schwab live chain GEX required" in reason
    assert SCHWAB_STALE_GEX_SOURCE in reason

    missing = live_gex_entry_block_reason({}, True)
    assert "current GEX source=missing" in missing


def test_fire_long_delta_proxy_allows_near_atm_missing_delta_only():
    ok, otm = fire_long_delta_proxy_ok("Bull Call Debit", 101.0, 100.0, 0.02)
    assert ok is True
    assert round(otm, 4) == 0.01

    ok, otm = fire_long_delta_proxy_ok("Bull Call Debit", 105.0, 100.0, 0.02)
    assert ok is False
    assert round(otm, 4) == 0.05

    ok, otm = fire_long_delta_proxy_ok("Bear Put Debit", 99.0, 100.0, 0.02)
    assert ok is True
    assert round(otm, 4) == 0.01

    ok, _ = fire_long_delta_proxy_ok("Iron Condor", 99.0, 100.0, 0.02)
    assert ok is False


def test_live_mode_date_violation_blocks_old_dated_folders_without_override():
    msg = live_mode_date_violation(
        dt.date(2026, 4, 23),
        dt.date(2026, 4, 26),
        historical_replay=False,
        allow_current_live=False,
    )
    assert "Refusing live-mode run for old dated folder 2026-04-23" in msg
    assert "--eod-live-planning" in msg

    assert (
        live_mode_date_violation(
            dt.date(2026, 4, 23),
            dt.date(2026, 4, 26),
            historical_replay=True,
            allow_current_live=False,
        )
        == ""
    )
    assert (
        live_mode_date_violation(
            dt.date(2026, 4, 23),
            dt.date(2026, 4, 26),
            historical_replay=False,
            allow_current_live=True,
        )
        == ""
    )
    assert (
        live_mode_date_violation(
            dt.date(2026, 4, 26),
            dt.date(2026, 4, 26),
            historical_replay=False,
            allow_current_live=False,
        )
        == ""
    )


def test_stage2_mode_name_distinguishes_eod_live_planning():
    assert stage2_mode_name(historical_replay=True, eod_live_planning=False) == "historical_replay"
    assert stage2_mode_name(historical_replay=False, eod_live_planning=True) == "eod_live_planning"
    assert stage2_mode_name(historical_replay=False, eod_live_planning=False) == "schwab_live"


def test_entry_gate_strict_pass_excludes_tolerated_debit_and_credit_misses():
    assert entry_gate_strict_pass("debit", 0.18, 0.18) is True
    assert entry_gate_strict_pass("debit", 0.29, 0.18) is False
    assert entry_gate_strict_pass("credit", 0.57, 0.57) is True
    assert entry_gate_strict_pass("credit", 0.48, 0.57) is False


def test_missing_gex_is_quality_blocker_not_hard_veto():
    hard, quality = split_approval_blockers(
        {
            "strategy": "Bull Call Debit",
            "approval_blockers": (
                "gex_missing;"
                "bull_call_missing_gex_without_uptrend:18.7<22.0;"
                "bull_call_rr_weak:1.7<2.0"
            ),
        }
    )

    assert "gex_missing" not in hard
    assert "bull_call_missing_gex_without_uptrend:18.7<22.0" not in hard
    assert "gex_missing" in quality
    assert "bull_call_missing_gex_without_uptrend:18.7<22.0" in quality
    assert "bull_call_rr_weak:1.7<2.0" in hard


def test_constructive_income_gex_can_downgrade_directional_ic_flow():
    hard, quality = split_approval_blockers(
        {
            "strategy": "Iron Condor",
            "gex_wall_context": "pinned_income_constructive",
            "iv_rank": 45.0,
            "approval_blockers": "contract_flow_directional;flow_too_directional_for_ic:bullish",
        }
    )

    assert hard == []
    assert "contract_flow_directional" in quality
    assert "flow_too_directional_for_ic:bullish" in quality


def test_normalize_probability_accepts_decimal_and_percent_scale():
    assert normalize_probability(0.286) == 0.286
    assert normalize_probability(28.6) == pytest.approx(0.286)
    assert normalize_probability("28.6%") == pytest.approx(0.286)


def test_pilot_convexity_policy_allows_weak_evidence_but_blocks_safety_failures():
    assert pilot_convexity_blockers_allow(
        [
            "stage1_conviction_below_yes_good:51<65",
            "stage1_flow_weak_or_ambiguous",
            "likelihood_verdict:LOW_SAMPLE",
            "signals_below:70<120",
            "fire_gex_pinned",
        ]
    )
    assert not pilot_convexity_blockers_allow(["live_entry_gate_fail"])
    assert not pilot_convexity_blockers_allow(["flow_not_confirmed:bullish/confirmed"])
    assert not pilot_convexity_blockers_allow(["contract_flow_contra"])
    assert not pilot_convexity_blockers_allow(["stage1_contract_flow_contra"])
    assert not pilot_convexity_blockers_allow(["pilot_ev_ml_below:0.599<1.5"])


def test_partial_ev_ml_debit_includes_spread_ramp():
    # BAC 51P/44P-style example: the binary shortcut is positive, but the
    # closed-form partial payoff is negative once the ramp zone is priced.
    ev = partial_ev_ml_debit(
        close=52.12,
        iv=0.250579,
        dte_days=14,
        long_strike=51,
        short_strike=44,
        net_debit=0.56,
        direction="bear",
    )
    assert ev == pytest.approx(-0.1357, abs=1e-3)
