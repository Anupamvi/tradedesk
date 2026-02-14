#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.trade_validator_pro", run_name="__main__")
else:
    from uwos.trade_validator_pro import *  # noqa: F401,F403
