#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.run_uw_deep_research_pipeline", run_name="__main__")
else:
    from uwos.run_uw_deep_research_pipeline import *  # noqa: F401,F403
