from scripts.update_latest_options_numbers import premium_income_for_row, premium_income_summary


def _row(*, opened, status, entry, realized, premium, open_pnl=0.0):
    return {
        "opened": opened,
        "status": status,
        "entry_cash_value": entry,
        "realized_pnl_value": realized,
        "premium_income_value": premium,
        "open_pnl_value": open_pnl,
    }


def test_open_credit_counts_entry_and_ignores_mark():
    value, rule = premium_income_for_row(entry_cash=500.0, realized_pnl=None, status="OPEN")

    assert value == 500.0
    assert "open credit" in rule


def test_closed_credit_counts_only_realized_profit_or_loss():
    profit, profit_rule = premium_income_for_row(
        entry_cash=4250.0,
        realized_pnl=250.0,
        status="CLOSED PROFIT",
    )
    loss, loss_rule = premium_income_for_row(
        entry_cash=1333.0,
        realized_pnl=-277.0,
        status="CLOSED LOSS",
    )

    assert profit == 250.0
    assert loss == -277.0
    assert "realized P/L" in profit_rule
    assert "realized P/L" in loss_rule


def test_open_debit_is_ignored_until_closed():
    open_value, _ = premium_income_for_row(entry_cash=-125.0, realized_pnl=None, status="OPEN")
    closed_value, _ = premium_income_for_row(
        entry_cash=-125.0,
        realized_pnl=75.0,
        status="CLOSED PROFIT",
    )

    assert open_value == 0.0
    assert closed_value == 75.0


def test_summary_separates_gross_credit_from_counted_pnl():
    rows = [
        _row(opened="2026-07-01", status="OPEN", entry=500.0, realized=0.0, premium=500.0),
        _row(opened="2026-07-02", status="CLOSED PROFIT", entry=4250.0, realized=250.0, premium=250.0),
        _row(opened="2026-07-03", status="CLOSED LOSS", entry=1333.0, realized=-277.0, premium=-277.0),
        _row(opened="2026-07-04", status="CLOSED PROFIT", entry=-125.0, realized=75.0, premium=75.0),
    ]

    summary = premium_income_summary(rows)["months"]["2026-07"]

    assert summary["gross_credit_collected"] == 6083.0
    assert summary["credit_income_counted"] == 500.0
    assert summary["closed_credit_profit_counted"] == 250.0
    assert summary["closed_debit_profit_counted"] == 75.0
    assert summary["closed_losses_counted"] == -277.0
    assert summary["premium_income_total"] == 548.0
