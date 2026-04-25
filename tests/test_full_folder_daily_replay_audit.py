import datetime as dt

from uwos.full_folder_daily_replay_audit import inventory


def _touch_family(day_dir, name):
    (day_dir / name).write_text("x", encoding="utf-8")


def test_inventory_requires_dp_and_bot_eod_as_separate_daily_families(tmp_path):
    complete = tmp_path / "2026-04-23"
    bot_only_for_dp = tmp_path / "2026-04-24"
    missing_bot = tmp_path / "2026-04-25"
    for day_dir in (complete, bot_only_for_dp, missing_bot):
        day_dir.mkdir()
        _touch_family(day_dir, f"stock-screener-{day_dir.name}.csv")
        _touch_family(day_dir, f"hot-chains-{day_dir.name}.csv")
        _touch_family(day_dir, f"chain-oi-changes-{day_dir.name}.csv")

    _touch_family(complete, "dp-eod-report-2026-04-23.csv")
    _touch_family(complete, "bot-eod-report-2026-04-23.zip")
    _touch_family(bot_only_for_dp, "bot-eod-report-2026-04-24.zip")
    _touch_family(missing_bot, "dp-eod-report-2026-04-25.csv")

    folders, incomplete = inventory(tmp_path, dt.date(2026, 4, 23), dt.date(2026, 4, 25))

    assert [folder.name for folder in folders] == ["2026-04-23"]
    missing_by_date = {row["date"]: row["missing"] for row in incomplete}
    assert missing_by_date["2026-04-24"] == ["dp"]
    assert missing_by_date["2026-04-25"] == ["bot_eod"]
