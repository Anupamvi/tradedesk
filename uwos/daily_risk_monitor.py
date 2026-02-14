#!/usr/bin/env python3
from __future__ import annotations

import sys

from uwos.report import main


if __name__ == "__main__":
    raise SystemExit(main(["daily", *sys.argv[1:]]))

