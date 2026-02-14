#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.eod_trade_scan_mode_a", run_name="__main__")
else:
    from uwos.eod_trade_scan_mode_a import *  # noqa: F401,F403
