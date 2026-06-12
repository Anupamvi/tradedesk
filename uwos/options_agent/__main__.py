"""CLI entrypoint for ``python3 -m uwos.options_agent``."""

from .core import main


if __name__ == "__main__":
    raise SystemExit(main())
