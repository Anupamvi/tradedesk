import pandas as pd
import pytest

from scripts.validate_strategy_universe import holm_adjust, validate


def test_validator_refuses_to_run_without_statistical_power() -> None:
    """Too few permutations floors the Holm p above threshold and rejects everything."""
    rows = []
    for sample, dates in (("TRAIN", range(25)), ("TEST", range(12))):
        for date in dates:
            for ticker, selected in (("A", True), ("B", False)):
                rows.append(
                    {
                        "signal_date": f"{sample}-{date:02d}",
                        "sector": "Technology",
                        "sector_state": "mixed",
                        "strategy": "long_call",
                        "ticker": f"{ticker}{date}",
                        "signal_selected": selected,
                        "sample": sample,
                        "pnl": 10.0,
                    }
                )
    with pytest.raises(SystemExit, match="zero power"):
        validate(pd.DataFrame(rows), permutations=20, bootstrap_trials=20)


def test_validator_rejects_signal_that_does_not_beat_matched_control() -> None:
    rows = []
    for sample, dates in (("TRAIN", range(25)), ("TEST", range(12))):
        for date in dates:
            for ticker, selected in (("A", True), ("B", False)):
                rows.append(
                    {
                        "signal_date": f"{sample}-{date:02d}",
                        "sector": "Technology",
                        "sector_state": "mixed",
                        "strategy": "long_call",
                        "ticker": f"{ticker}{date}",
                        "signal_selected": selected,
                        "sample": sample,
                        "pnl": 10.0,
                    }
                )
    detail, _ = validate(pd.DataFrame(rows), permutations=100, bootstrap_trials=20)
    lane = detail[(detail["scope"] == "sector") & (detail["scope_value"] == "Technology")].iloc[0]

    assert not lane["screen_pass"]
    assert lane["release_status"] == "REJECTED"


def test_holm_adjustment_uses_full_hypothesis_family() -> None:
    adjusted = holm_adjust(pd.Series([0.001, 0.02, 0.50]))

    assert adjusted.tolist() == [0.003, 0.04, 0.5]