#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.strategy_engine", run_name="__main__")
else:
    from uwos.strategy_engine import *  # noqa: F401,F403
