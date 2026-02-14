from __future__ import annotations

import sys

from .report import main


def run(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    return main(["weekly", *args])


if __name__ == "__main__":
    raise SystemExit(run())
