#!/usr/bin/env python3
"""CLI wrapper for the Options Agent market-open rerun guard."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uwos.options_agent.market_open_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
