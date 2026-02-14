#!/usr/bin/env python3
from __future__ import annotations

if __name__ == "__main__":
    import runpy

    runpy.run_module("uwos.uw_os_v6_fullpack_fixed", run_name="__main__")
else:
    from uwos.uw_os_v6_fullpack_fixed import *  # noqa: F401,F403
