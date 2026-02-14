#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.validate_schwab_full_year", run_name="__main__")
else:
    from uwos.validate_schwab_full_year import *  # noqa: F401,F403
