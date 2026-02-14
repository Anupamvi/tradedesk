#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.run_trade_playbook", run_name="__main__")
else:
    from uwos.run_trade_playbook import *  # noqa: F401,F403
