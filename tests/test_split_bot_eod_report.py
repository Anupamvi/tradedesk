import csv
import zipfile

from uwos.split_bot_eod_report import split_bot_eod_report


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
