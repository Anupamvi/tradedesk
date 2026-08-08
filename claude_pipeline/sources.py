"""Discovery and resolution of the five Unusual Whales daily exports.

Every file is keyed by the date in its FILENAME, never by the folder it sits in:
folders have been observed to contain another session's complete export.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

UW_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")

FAMILIES: tuple[str, ...] = (
    "stock-screener",
    "hot-chains",
    "chain-oi-changes",
    "bot-eod-report",
    "dp-eod-report",
)

SESSION_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Trailing part of a filename once the family prefix is stripped, e.g.
#   "-2026-08-06.zip"                  -> exact export
#   "-latest-2026-05-04.zip"           -> a LATER session's open interest
#   "-2026-04-23.part-01-of-05.zip"    -> one piece of a split export
#   "-2025-12-19 (2).zip"              -> browser re-download; sometimes the ONLY copy
_SUFFIX_RE = re.compile(
    r"^-(?:(?P<qualifier>[a-z]+)-)?(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:\.part-(?P<part>\d+)-of-(?P<parts>\d+))?"
    r"(?P<copy> \(\d+\))?\.zip$"
)

# These qualifiers name a session AFTER the folder they sit in; the same data
# always arrives again in its own dated folder, so they are never worth the risk.
FORWARD_QUALIFIERS = frozenset({"latest", "current"})


@dataclass(frozen=True)
class SourceFile:
    path: Path
    family: str
    session: str
    part: int | None = None
    part_count: int | None = None
    is_copy: bool = False


@dataclass
class Rejection:
    path: Path
    reason: str


@dataclass
class SourceIndex:
    files: dict[tuple[str, str], list[SourceFile]] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)

    def sessions(self) -> list[str]:
        return sorted({session for session, _ in self.files})

    def families_for(self, session: str) -> set[str]:
        return {fam for sess, fam in self.files if sess == session}

    def complete_sessions(self, families: tuple[str, ...] = FAMILIES) -> list[str]:
        return [s for s in self.sessions() if set(families) <= self.families_for(s)]

    def get(self, session: str, family: str) -> list[SourceFile]:
        found = self.files.get((session, family), [])
        # Cheap standing guard against the class of bug this module exists to prevent.
        assert all(f.session == session for f in found), "index returned a foreign session"
        return found


def _classify(path: Path) -> tuple[SourceFile | None, str | None]:
    name = path.name
    family = next((f for f in sorted(FAMILIES, key=len, reverse=True) if name.startswith(f)), None)
    if family is None:
        return None, "not_a_known_family"
    match = _SUFFIX_RE.match(name[len(family):])
    if match is None:
        return None, "unparsed_filename"
    qualifier = match.group("qualifier")
    if qualifier in FORWARD_QUALIFIERS:
        return None, f"forward_dated_{qualifier}_export"
    if qualifier is not None:
        return None, f"unknown_qualifier_{qualifier}"
    part = match.group("part")
    return (
        SourceFile(
            path=path,
            family=family,
            session=match.group("date"),
            part=int(part) if part else None,
            part_count=int(match.group("parts")) if part else None,
            is_copy=match.group("copy") is not None,
        ),
        None,
    )


def build_index(root: Path = UW_ROOT) -> SourceIndex:
    grouped: dict[tuple[str, str], list[SourceFile]] = defaultdict(list)
    index = SourceIndex()

    for day_dir in sorted(d for d in root.iterdir() if d.is_dir() and SESSION_DIR_RE.match(d.name)):
        for path in sorted(day_dir.glob("*.zip")):
            source, reason = _classify(path)
            if source is None:
                index.rejections.append(Rejection(path, reason or "unknown"))
                continue
            grouped[(source.session, source.family)].append(source)

    for key, found in grouped.items():
        parts = [f for f in found if f.part is not None]
        whole = [f for f in found if f.part is None]

        if parts:
            expected = parts[0].part_count or 0
            seen = {f.part for f in parts}
            if len(({*range(1, expected + 1)} - seen)) > 0:
                index.rejections.append(
                    Rejection(parts[0].path, f"incomplete_split_export_{len(seen)}_of_{expected}")
                )
                continue
            index.files[key] = sorted(parts, key=lambda f: f.part or 0)
            continue

        if len(whole) > 1:
            # Prefer a canonical name over a re-download copy, and a file sitting in
            # its own session folder over the same export filed under another day.
            whole.sort(key=lambda f: (f.is_copy, f.path.parent.name != f.session, f.path.name))
            for extra in whole[1:]:
                index.rejections.append(Rejection(extra.path, "duplicate_export_for_session"))
        index.files[key] = whole[:1]

    return index
