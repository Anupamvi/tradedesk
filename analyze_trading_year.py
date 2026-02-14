#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.analyze_trading_year", run_name="__main__")
else:
    from uwos.analyze_trading_year import *  # noqa: F401,F403
