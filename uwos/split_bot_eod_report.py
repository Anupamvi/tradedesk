#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from uwos.whale_source import BOT_EOD_PREFIX, infer_date_from_path, open_bot_eod


csv.field_size_limit(sys.maxsize)


@dataclass
class SplitPart:
    index: int
    path: str
    csv_name: str
    rows: int


@dataclass
class SplitResult:
    source_path: str
    source_label: str
    total_rows: int
    parts_requested: int
    output_format: str
    manifest_path: str
    parts: list[SplitPart]


def _report_stem(input_path: Path) -> str:
    stem = input_path.name
    for suffix in (".zip", ".csv"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.startswith(BOT_EOD_PREFIX):
        return stem
    date_str = infer_date_from_path(input_path)
    if date_str != "Unknown Date":
        return f"{BOT_EOD_PREFIX}{date_str}"
    return input_path.stem


def _part_row_counts(total_rows: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total_rows // parts
    remainder = total_rows % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def _read_header_and_count(input_path: Path) -> tuple[list[str], int, str]:
    with open_bot_eod(input_path) as (handle, source_label):
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input report is empty: {input_path}") from exc
        total_rows = sum(1 for _ in reader)
    return header, total_rows, source_label


@contextmanager
def _open_part_writer(path: Path, csv_name: str, output_format: str) -> Iterator[csv.writer]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            yield csv.writer(handle)
        return

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        with zf.open(csv_name, "w") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            try:
                yield csv.writer(text)
            finally:
                text.flush()
                text.detach()


def split_bot_eod_report(
    input_path: Path,
    *,
    out_dir: Path | None = None,
    parts: int = 5,
    output_format: str = "zip",
    overwrite: bool = False,
) -> SplitResult:
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input report: {input_path}")
    if output_format not in {"zip", "csv"}:
        raise ValueError("output_format must be 'zip' or 'csv'")
    if parts <= 0:
        raise ValueError("parts must be positive")

    out_dir = (Path(out_dir).expanduser().resolve() if out_dir else input_path.parent)
    report_stem = _report_stem(input_path)
    header, total_rows, source_label = _read_header_and_count(input_path)
    row_counts = _part_row_counts(total_rows, parts)

    planned: list[tuple[Path, str, int]] = []
    for idx, rows_for_part in enumerate(row_counts, start=1):
        part_stem = f"{report_stem}.part-{idx:02d}-of-{parts:02d}"
        csv_name = f"{part_stem}.csv"
        suffix = ".zip" if output_format == "zip" else ".csv"
        out_path = out_dir / f"{part_stem}{suffix}"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
        planned.append((out_path, csv_name, rows_for_part))

    written: list[SplitPart] = []
    with open_bot_eod(input_path) as (handle, _source_label):
        reader = csv.reader(handle)
        next(reader)
        for idx, (out_path, csv_name, rows_for_part) in enumerate(planned, start=1):
            rows_written = 0
            with _open_part_writer(out_path, csv_name, output_format) as writer:
                writer.writerow(header)
                for _ in range(rows_for_part):
                    try:
                        row = next(reader)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Input ended early while writing part {idx}; source may have changed during split."
                        ) from exc
                    writer.writerow(row)
                    rows_written += 1
            written.append(
                SplitPart(
                    index=idx,
                    path=str(out_path),
                    csv_name=csv_name,
                    rows=rows_written,
                )
            )

    if sum(part.rows for part in written) != total_rows:
        raise RuntimeError("Split row count mismatch after writing parts")

    manifest_path = out_dir / f"{report_stem}.split-manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path} (use --overwrite)")

    result = SplitResult(
        source_path=str(input_path),
        source_label=source_label,
        total_rows=total_rows,
        parts_requested=parts,
        output_format=output_format,
        manifest_path=str(manifest_path),
        parts=written,
    )
    manifest_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a large bot-eod-report-YYYY-MM-DD CSV/ZIP into smaller row-balanced files."
    )
    parser.add_argument("input", type=Path, help="Path to bot-eod-report-YYYY-MM-DD.csv or .zip")
    parser.add_argument(
        "--parts",
        type=int,
        default=5,
        help="Number of output files to create. Default: 5.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as input.",
    )
    parser.add_argument(
        "--output-format",
        choices=["zip", "csv"],
        default="zip",
        help="Write each part as a compressed ZIP containing one CSV, or as plain CSV. Default: zip.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing part files and manifest.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = split_bot_eod_report(
        args.input,
        out_dir=args.out_dir,
        parts=args.parts,
        output_format=args.output_format,
        overwrite=bool(args.overwrite),
    )
    rows_per_part = ", ".join(str(part.rows) for part in result.parts)
    print(f"Source rows: {result.total_rows:,}")
    print(f"Wrote {len(result.parts)} {result.output_format} part files")
    print(f"Rows per part: {rows_per_part}")
    print(f"Manifest: {result.manifest_path}")
    for part in result.parts:
        print(f"  part {part.index:02d}: {part.rows:,} rows -> {part.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
