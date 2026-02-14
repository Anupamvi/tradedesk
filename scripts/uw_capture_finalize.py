#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.uw_capture_finalize", run_name="__main__")
else:
    from uwos.uw_capture_finalize import *  # noqa: F401,F403
