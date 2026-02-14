#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.daily_risk_monitor", run_name="__main__")
else:
    from uwos.daily_risk_monitor import *  # noqa: F401,F403
