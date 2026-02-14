#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.setup_likelihood_backtest", run_name="__main__")
else:
    from uwos.setup_likelihood_backtest import *  # noqa: F401,F403
