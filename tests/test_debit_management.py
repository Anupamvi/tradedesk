from uwos.debit_management import evaluate_debit_management


def test_near_otm_with_time_and_supportive_trend_holds():
    verdict, reason = evaluate_debit_management(
        otm_pct=2.47,
        dte=27,
        loss_pct=1.52,
        trend_against_thesis=False,
    )
    assert verdict == "HOLD"
    assert "trend supports" in reason


def test_mid_otm_requires_adverse_trend_to_escalate():
    assert evaluate_debit_management(otm_pct=4.0, dte=25, loss_pct=10, trend_against_thesis=False)[0] == "HOLD"
    assert evaluate_debit_management(otm_pct=4.0, dte=25, loss_pct=10, trend_against_thesis=True)[0] == "ASSESS"
    assert evaluate_debit_management(otm_pct=4.0, dte=25, loss_pct=10, trend_against_thesis=None)[0] == "HOLD"


def test_loss_and_expiry_boundaries_escalate():
    assert evaluate_debit_management(otm_pct=1, dte=25, loss_pct=40, trend_against_thesis=False)[0] == "ASSESS"
    assert evaluate_debit_management(otm_pct=1, dte=25, loss_pct=60, trend_against_thesis=False)[0] == "CLOSE"
    assert evaluate_debit_management(otm_pct=6, dte=34, loss_pct=5, trend_against_thesis=False)[0] == "CLOSE"
    assert evaluate_debit_management(otm_pct=1, dte=14, loss_pct=5, trend_against_thesis=False)[0] == "CLOSE"
