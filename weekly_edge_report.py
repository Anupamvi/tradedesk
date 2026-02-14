#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.weekly_edge_report", run_name="__main__")
else:
    from uwos.weekly_edge_report import *  # noqa: F401,F403
