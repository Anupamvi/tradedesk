"""CLI entrypoint for ``python3 -m uwos.lessonengine``."""

from .core import main


if __name__ == "__main__":
    raise SystemExit(main())
