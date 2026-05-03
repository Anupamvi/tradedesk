import csv
import subprocess
import sys
import zipfile
from pathlib import Path

from uwos.split_bot_eod_report import main, resolve_input_and_out_dir, split_bot_eod_report


def _write_source_zip(path, rows):
    csv_name = path.with_suffix(".csv").name
    with zipfile.ZipFile(path, "w") as zf:
        with zf.open(csv_name, "w") as raw:
            text = raw
            payload = "underlying_symbol,premium\n" + "\n".join(f"T{i},{i}" for i in rows) + "\n"
            text.write(payload.encode("utf-8"))


def _read_part_zip(path):
    with zipfile.ZipFile(path) as zf:
        [name] = zf.namelist()
        with zf.open(name) as raw:
            return list(csv.reader(line.decode("utf-8") for line in raw.readlines()))


def test_split_bot_eod_report_writes_five_balanced_zip_parts(tmp_path):
    source = tmp_path / "bot-eod-report-2026-04-23.zip"
    _write_source_zip(source, range(11))

    result = split_bot_eod_report(source, out_dir=tmp_path / "parts")

    assert result.total_rows == 11
    assert [part.rows for part in result.parts] == [3, 2, 2, 2, 2]
    assert len(result.parts) == 5
    assert (tmp_path / "parts" / "bot-eod-report-2026-04-23.split-manifest.json").exists()

    seen = []
    for part in result.parts:
        rows = _read_part_zip(part.path)
        assert rows[0] == ["underlying_symbol", "premium"]
        seen.extend(rows[1:])

    assert seen == [[f"T{i}", str(i)] for i in range(11)]


def test_resolve_input_and_out_dir_accepts_date(tmp_path):
    day_dir = tmp_path / "2026-04-26"
    day_dir.mkdir()
    source = day_dir / "bot-eod-report-2026-04-26.zip"
    _write_source_zip(source, range(3))

    input_path, out_dir = resolve_input_and_out_dir("2026-04-26", root_dir=tmp_path)

    assert input_path == source
    assert out_dir == day_dir


def test_main_accepts_date_and_writes_parts_in_date_folder(tmp_path):
    day_dir = tmp_path / "2026-04-26"
    day_dir.mkdir()
    _write_source_zip(day_dir / "bot-eod-report-2026-04-26.zip", range(6))

    assert main(["2026-04-26", "--root-dir", str(tmp_path), "--parts", "5"]) == 0

    assert len(list(day_dir.glob("bot-eod-report-2026-04-26.part-*-of-05.zip"))) == 5
    assert (day_dir / "bot-eod-report-2026-04-26.split-manifest.json").exists()


def test_script_runs_directly_from_uwos_directory():
    script = Path(__file__).resolve().parents[1] / "uwos" / "split_bot_eod_report.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(script.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    assert "Split a large bot-eod-report" in proc.stdout
