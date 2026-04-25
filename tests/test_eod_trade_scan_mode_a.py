from pathlib import Path

from uwos.eod_trade_scan_mode_a import expected_input_date, find_asof, non_bot_input_zips, pick_csv_map_entry


def test_non_bot_input_zips_excludes_bot_eod_from_asof_detection(tmp_path):
    (tmp_path / "chain-oi-changes-2026-04-23.zip").write_text("x", encoding="utf-8")
    (tmp_path / "dp-eod-report-2026-04-23.zip").write_text("x", encoding="utf-8")
    (tmp_path / "hot-chains-2026-04-23.zip").write_text("x", encoding="utf-8")
    (tmp_path / "stock-screener-2026-04-23.zip").write_text("x", encoding="utf-8")
    (tmp_path / "bot-eod-report-2026-04-24.zip").write_text("x", encoding="utf-8")

    zips = non_bot_input_zips(tmp_path)

    assert [path.name for path in zips] == [
        "chain-oi-changes-2026-04-23.zip",
        "dp-eod-report-2026-04-23.zip",
        "hot-chains-2026-04-23.zip",
        "stock-screener-2026-04-23.zip",
    ]
    assert find_asof(zips) == "2026-04-23"


def test_pick_csv_map_entry_requires_matching_asof_for_prefix():
    csv_map = {
        "chain-oi-changes-2026-04-22.zip": Path("chain-oi-changes-2026-04-22.csv"),
        "chain-oi-changes-2026-04-23.zip": Path("chain-oi-changes-2026-04-23.csv"),
    }

    zname, cpath = pick_csv_map_entry(csv_map, "chain-oi-changes-", "2026-04-23")

    assert zname == "chain-oi-changes-2026-04-23.zip"
    assert cpath.name == "chain-oi-changes-2026-04-23.csv"


def test_expected_input_date_uses_dated_folder_name(tmp_path):
    day_dir = tmp_path / "2026-04-23"
    day_dir.mkdir()

    assert expected_input_date(day_dir) == "2026-04-23"
