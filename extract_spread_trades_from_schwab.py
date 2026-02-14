#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.extract_spread_trades_from_schwab", run_name="__main__")
else:
    from uwos.extract_spread_trades_from_schwab import *  # noqa: F401,F403
