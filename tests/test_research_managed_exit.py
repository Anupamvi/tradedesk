from scripts.managed_exit_backtest import has_full_observation_horizon


def test_full_observation_horizon_includes_entry_and_time_stop() -> None:
    sessions = [f"D{index}" for index in range(42)]

    assert has_full_observation_horizon(0, sessions, max_hold=40)
    assert not has_full_observation_horizon(1, sessions, max_hold=40)