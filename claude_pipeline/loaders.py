"""Reading the Unusual Whales exports resolved by :mod:`claude_pipeline.sources`."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from claude_pipeline.sources import SourceFile, SourceIndex


class MissingSourceError(FileNotFoundError):
    pass


def _csv_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
    if not members:
        raise MissingSourceError("archive contains no csv member")
    return members[0]


def _read_bytes(source: SourceFile) -> bytes:
    """Return the CSV bytes, unwrapping one level of zip-inside-zip bundling.

    One export on disk is a bundle of several sessions' archives; only the member
    naming the requested session may be used.
    """
    with zipfile.ZipFile(source.path) as archive:
        try:
            member = _csv_member(archive)
        except MissingSourceError:
            inner = [
                m for m in archive.infolist()
                if m.filename.lower().endswith(".zip") and source.session in m.filename
            ]
            if not inner:
                raise MissingSourceError(
                    f"{source.path.name} has no csv and no nested archive for {source.session}"
                ) from None
            with archive.open(inner[0]) as handle:
                nested_bytes = handle.read()
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                with nested.open(_csv_member(nested)) as handle:
                    return handle.read()
        with archive.open(member) as handle:
            return handle.read()


def _resolve(index: SourceIndex, session: str, family: str) -> list[SourceFile]:
    found = index.get(session, family)
    if not found:
        raise MissingSourceError(f"no {family} export for session {session}")
    return found


def read(
    index: SourceIndex,
    session: str,
    family: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one family for one session, concatenating any split parts."""
    frames = [
        pd.read_csv(io.BytesIO(_read_bytes(source)), usecols=columns, low_memory=False)
        for source in _resolve(index, session, family)
    ]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def iter_chunks(
    index: SourceIndex,
    session: str,
    family: str,
    columns: list[str] | None = None,
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    """Stream a family in chunks. Required for the multi-gigabyte trade tape."""
    for source in _resolve(index, session, family):
        with zipfile.ZipFile(source.path) as archive:
            try:
                member = _csv_member(archive)
            except MissingSourceError:
                yield from pd.read_csv(
                    io.BytesIO(_read_bytes(source)), usecols=columns,
                    chunksize=chunksize, low_memory=False,
                )
                continue
            with archive.open(member) as handle:
                yield from pd.read_csv(
                    handle, usecols=columns, chunksize=chunksize, low_memory=False
                )


def count_rows(index: SourceIndex, session: str, family: str) -> int:
    total = 0
    for source in _resolve(index, session, family):
        with zipfile.ZipFile(source.path) as archive:
            try:
                member = _csv_member(archive)
            except MissingSourceError:
                total += _read_bytes(source).count(b"\n") - 1
                continue
            with archive.open(member) as handle:
                while chunk := handle.read(16 << 20):
                    total += chunk.count(b"\n")
            total -= 1  # each part carries its own header row
    return total


def uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(m.file_size for m in archive.infolist())
