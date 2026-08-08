"""Stage 1 proof: what source discovery finds, and everything it refused."""

from __future__ import annotations

from collections import Counter

from claude_pipeline.sources import FAMILIES, SESSION_DIR_RE, UW_ROOT, build_index


def main() -> None:
    index = build_index()
    sessions = index.sessions()
    folders = sorted(d.name for d in UW_ROOT.iterdir() if d.is_dir() and SESSION_DIR_RE.match(d.name))

    print(f"dated folders on disk : {len(folders)}")
    print(f"sessions indexed      : {len(sessions)}  ({sessions[0]} -> {sessions[-1]})")

    print("\ncoverage by family:")
    for family in FAMILIES:
        have = [s for s in sessions if index.get(s, family)]
        parts = sum(len(index.get(s, family)) for s in have)
        print(f"  {family:<20} {len(have):>4} sessions   {parts:>4} files")
    complete = index.complete_sessions()
    print(f"  {'ALL FIVE':<20} {len(complete):>4} sessions   first={complete[0]}")

    print("\nrejected files:")
    for reason, count in Counter(r.reason for r in index.rejections).most_common():
        print(f"  {count:>4}  {reason}")

    print("\nsessions whose files live in a differently-named folder:")
    misfiled = [
        (session, family, file.path.parent.name)
        for (session, family), found in sorted(index.files.items())
        for file in found
        if file.path.parent.name != session
    ]
    for session, family, folder in misfiled:
        print(f"  session {session}  {family:<18} found in folder {folder}")
    if not misfiled:
        print("  none")

    print("\nsessions per month with all five families:")
    by_month = Counter(s[:7] for s in complete)
    total = Counter(s[:7] for s in sessions)
    for month in sorted(total):
        print(f"  {month}: {by_month.get(month, 0):>3} of {total[month]:>3} indexed sessions")


if __name__ == "__main__":
    main()
