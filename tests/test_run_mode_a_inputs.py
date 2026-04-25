import os
import time
import zipfile

from uwos.run_mode_a_two_stage import pick_csvs


def _write_zip(day_dir, name):
    csv_name = name.replace(".zip", ".csv")
    zip_path = day_dir / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, "col\n1\n")
    return zip_path


def test_pick_csvs_reextracts_stale_cache_from_wrong_date(tmp_path):
    day_dir = tmp_path / "2026-04-23"
    day_dir.mkdir()
    for prefix in ("chain-oi-changes-", "dp-eod-report-", "hot-chains-", "stock-screener-"):
        zip_path = _write_zip(day_dir, f"{prefix}2026-04-23.zip")
        old = time.time() - 60
        os.utime(zip_path, (old, old))

    cache = day_dir / "_unzipped_mode_a"
    cache.mkdir()
    for prefix in ("chain-oi-changes-", "dp-eod-report-", "hot-chains-", "stock-screener-"):
        (cache / f"{prefix}2026-04-22.csv").write_text("stale\n1\n", encoding="utf-8")

    selected = pick_csvs(day_dir)

    assert {path.name for path in selected.values()} == {
        "chain-oi-changes-2026-04-23.csv",
        "dp-eod-report-2026-04-23.csv",
        "hot-chains-2026-04-23.csv",
        "stock-screener-2026-04-23.csv",
    }
