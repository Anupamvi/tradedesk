import datetime as dt

from multi_strategy_backtest import evaluation_schedule


def test_eod_signal_enters_next_session_and_exits_after_entry():
    dates = [dt.date(2026, 1, day) for day in range(2, 12)]

    entry_day, exit_days = evaluation_schedule(dates, 0, horizon=5)

    assert entry_day == dates[1]
    assert exit_days == dates[2:7]
    assert dates[0] not in [entry_day, *exit_days]


def test_evaluation_schedule_rejects_incomplete_post_entry_horizon():
    dates = [dt.date(2026, 1, day) for day in range(2, 8)]

    assert evaluation_schedule(dates, 0, horizon=5) is None
